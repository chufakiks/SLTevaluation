#!/usr/bin/env python3
"""
Batch inference script for SignWriting translation pipeline.

Processes videos from a CSV file and outputs translations.

Usage:
    python batch_inference_signwriting.py --csv input.csv --output results.csv

CSV format (input):
    video_id
    video1
    video2

CSV format (output):
    video_id,video_path,signs_detected,signwriting,translation,full_text,duration,fps,
    time_total_ms,time_pose_extraction_ms,time_segmentation_ms,time_transcription_ms,
    time_translation_ms,status,error
"""

import argparse
import csv
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
import torch
from tqdm import tqdm

# pose-format imports
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.utils.holistic import holistic_components
from pose_format.utils.generic import reduce_holistic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Stores timing information for a single video processing run."""
    total_ms: float = 0.0
    pose_extraction_ms: float = 0.0
    segmentation_ms: float = 0.0
    transcription_ms: float = 0.0
    translation_ms: float = 0.0


class CUDATimer:
    """
    GPU-accurate timer using CUDA events.

    Falls back to CPU timing if CUDA is not available.
    """

    def __init__(self, device: str = "cuda"):
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self._start_event = None
        self._end_event = None
        self._start_time = None

    def start(self):
        """Start the timer."""
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds."""
        if self.use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event)
        else:
            return (time.perf_counter() - self._start_time) * 1000.0


class TimedSection:
    """Context manager for timing code sections with CUDA events."""

    def __init__(self, timer: CUDATimer):
        self.timer = timer
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.timer.start()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = self.timer.stop()


class SignWritingBatchInference:
    """
    Batch inference pipeline for SignWriting translation.

    Pipeline: Video -> MediaPipe Holistic -> Pose -> Segmentation -> SignWriting -> Text
    """

    def __init__(
        self,
        experiment_dir: str = "experiment",
        model_name: str = "bc2de71.ckpt",
        target_language: str = "en",
        device: str = "cuda"
    ):
        """
        Initialize the pipeline.

        Args:
            experiment_dir: Directory containing model files
            model_name: SignWriting transcription model checkpoint
            target_language: Target language for translation (e.g., "en", "de")
            device: Device to use ("cuda" or "cpu")
        """
        self.experiment_dir = Path(experiment_dir)
        self.model_name = model_name
        self.target_language = target_language

        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            if device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")

        # Lazy-loaded components
        self._holistic = None
        self._translator = None
        self._sw_tokenizer = None

        # GPU-accurate timer
        self.timer = CUDATimer(self.device)

        logger.info(f"Initializing SignWriting pipeline")
        logger.info(f"  Experiment dir: {self.experiment_dir}")
        logger.info(f"  Target language: {self.target_language}")
        logger.info(f"  Device: {self.device}")

    def _init_mediapipe(self):
        """Initialize MediaPipe Holistic model."""
        if self._holistic is None:
            logger.info("Loading MediaPipe Holistic model...")
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._holistic

    def _init_translator(self):
        """Initialize SignWriting to text translator."""
        if self._translator is None:
            logger.info("Loading SignWriting translator...")
            from signwriting.tokenizer import SignWritingTokenizer
            from signwriting_translation.bin import load_sockeye_translator

            self._sw_tokenizer = SignWritingTokenizer()
            self._translator, _ = load_sockeye_translator(
                "sign/sockeye-signwriting-to-text",
                log_timing=False
            )
        return self._translator, self._sw_tokenizer

    def _download_transcription_model(self):
        """Download the SignWriting transcription model if needed."""
        from signwriting_transcription.pose_to_signwriting.bin import download_model

        self.experiment_dir.mkdir(exist_ok=True)
        download_model(self.experiment_dir, self.model_name)
        logger.info(f"Transcription model ready at {self.experiment_dir}")

    def extract_poses_from_video(self, video_path: str) -> tuple[list[dict], int, int, float]:
        """
        Extract MediaPipe Holistic poses from a video file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (frames_data, width, height, fps)
        """
        holistic = self._init_mediapipe()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        frames_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            frame_dict = {
                'poseLandmarks': self._landmarks_to_list(results.pose_landmarks, 33),
                'faceLandmarks': self._landmarks_to_list(results.face_landmarks, 468),
                'leftHandLandmarks': self._landmarks_to_list(results.left_hand_landmarks, 21),
                'rightHandLandmarks': self._landmarks_to_list(results.right_hand_landmarks, 21),
            }
            frames_data.append(frame_dict)

        cap.release()
        logger.info(f"Extracted {len(frames_data)} frames")

        return frames_data, width, height, fps

    def _landmarks_to_list(self, landmarks, expected_count: int) -> Optional[list[dict]]:
        """Convert MediaPipe landmarks to list of dicts."""
        if landmarks is None:
            return None

        result = []
        for lm in landmarks.landmark[:expected_count]:
            result.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': getattr(lm, 'visibility', 1.0)
            })
        return result

    def frames_to_pose(
        self,
        frames: list[dict],
        width: int,
        height: int,
        fps: float
    ) -> Pose:
        """
        Convert extracted frames to a Pose object.

        Args:
            frames: List of frame dicts with landmarks
            width: Video width
            height: Video height
            fps: Frames per second

        Returns:
            Pose object
        """
        num_frames = len(frames)

        # Standard MediaPipe Holistic point counts
        POSE_POINTS = 33
        FACE_POINTS = 468
        HAND_POINTS = 21
        TOTAL_POINTS = POSE_POINTS + FACE_POINTS + HAND_POINTS * 2  # 543

        # Initialize arrays
        data = np.zeros((num_frames, 1, TOTAL_POINTS, 3), dtype=np.float32)
        confidence = np.zeros((num_frames, 1, TOTAL_POINTS), dtype=np.float32)

        for frame_idx, frame in enumerate(frames):
            point_offset = 0

            # POSE_LANDMARKS (33 points)
            pose_landmarks = frame.get('poseLandmarks') or []
            for i, lm in enumerate(pose_landmarks[:POSE_POINTS]):
                if lm:
                    data[frame_idx, 0, point_offset + i, 0] = lm.get('x', 0) * width
                    data[frame_idx, 0, point_offset + i, 1] = lm.get('y', 0) * height
                    data[frame_idx, 0, point_offset + i, 2] = lm.get('z', 0) * width
                    confidence[frame_idx, 0, point_offset + i] = lm.get('visibility', 1.0)
            point_offset += POSE_POINTS

            # FACE_LANDMARKS (468 points)
            face_landmarks = frame.get('faceLandmarks') or []
            for i, lm in enumerate(face_landmarks[:FACE_POINTS]):
                if lm:
                    data[frame_idx, 0, point_offset + i, 0] = lm.get('x', 0) * width
                    data[frame_idx, 0, point_offset + i, 1] = lm.get('y', 0) * height
                    data[frame_idx, 0, point_offset + i, 2] = lm.get('z', 0) * width
                    confidence[frame_idx, 0, point_offset + i] = lm.get('visibility', 1.0)
            point_offset += FACE_POINTS

            # LEFT_HAND_LANDMARKS (21 points)
            left_hand = frame.get('leftHandLandmarks') or []
            for i, lm in enumerate(left_hand[:HAND_POINTS]):
                if lm:
                    data[frame_idx, 0, point_offset + i, 0] = lm.get('x', 0) * width
                    data[frame_idx, 0, point_offset + i, 1] = lm.get('y', 0) * height
                    data[frame_idx, 0, point_offset + i, 2] = lm.get('z', 0) * width
                    confidence[frame_idx, 0, point_offset + i] = lm.get('visibility', 1.0)
            point_offset += HAND_POINTS

            # RIGHT_HAND_LANDMARKS (21 points)
            right_hand = frame.get('rightHandLandmarks') or []
            for i, lm in enumerate(right_hand[:HAND_POINTS]):
                if lm:
                    data[frame_idx, 0, point_offset + i, 0] = lm.get('x', 0) * width
                    data[frame_idx, 0, point_offset + i, 1] = lm.get('y', 0) * height
                    data[frame_idx, 0, point_offset + i, 2] = lm.get('z', 0) * width
                    confidence[frame_idx, 0, point_offset + i] = lm.get('visibility', 1.0)

        # Create header
        dimensions = PoseHeaderDimensions(width=width, height=height, depth=0)
        components = holistic_components()
        header = PoseHeader(version=0.2, dimensions=dimensions, components=components)

        # Create body
        body = NumPyPoseBody(fps=fps, data=data, confidence=confidence)

        return Pose(header=header, body=body)

    def segment_pose(self, pose: Pose) -> tuple[list[dict], list[dict]]:
        """
        Run sign language segmentation on pose data.

        Args:
            pose: Pose object

        Returns:
            Tuple of (sign_segments, sentence_segments)
        """
        from sign_language_segmentation.bin import segment_pose as _segment_pose

        logger.info("Running segmentation...")
        _, tiers = _segment_pose(pose, verbose=False)

        sign_segments = tiers.get("SIGN", [])
        sentence_segments = tiers.get("SENTENCE", [])

        logger.info(f"Detected {len(sign_segments)} signs, {len(sentence_segments)} sentences")

        return sign_segments, sentence_segments

    def transcribe_to_signwriting(
        self,
        pose: Pose,
        sign_segments: list[dict],
        fps: float
    ) -> list[str]:
        """
        Transcribe pose segments to SignWriting notation.

        Args:
            pose: Pose object
            sign_segments: List of sign segment dicts with 'start' and 'end' keys
            fps: Frames per second

        Returns:
            List of SignWriting FSW strings
        """
        if not sign_segments:
            return []

        from signwriting_transcription.pose_to_signwriting.bin import preprocessing_signs
        from signwriting_transcription.pose_to_signwriting.joeynmt_pose.prediction import (
            translate as sw_translate
        )

        # Ensure model is downloaded
        self._download_transcription_model()

        # Preprocess pose (reduce holistic)
        preprocessed_pose = reduce_holistic(pose)

        # Convert segments to millisecond format expected by preprocessing
        sign_annotations = []
        for seg in sign_segments:
            start_frame = seg.get('start', seg[0] if isinstance(seg, (list, tuple)) else 0)
            end_frame = seg.get('end', seg[1] if isinstance(seg, (list, tuple)) else 0)
            start_ms = int(start_frame / fps * 1000)
            end_ms = int(end_frame / fps * 1000)
            sign_annotations.append((start_ms, end_ms, ""))

        logger.info(f"Transcribing {len(sign_annotations)} signs to SignWriting...")

        # Run transcription
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_files = preprocessing_signs(
                preprocessed_pose,
                sign_annotations,
                'tight',
                temp_dir
            )
            config_path = str(self.experiment_dir / "config.yaml")
            signwriting_list = sw_translate(config_path, temp_files)

        logger.info(f"Transcription complete: {signwriting_list}")
        return signwriting_list

    def translate_signwriting_to_text(self, signwriting_list: list[str]) -> list[str]:
        """
        Translate SignWriting notation to text.

        Args:
            signwriting_list: List of SignWriting FSW strings

        Returns:
            List of translated text strings
        """
        if not signwriting_list:
            return []

        from signwriting_translation.bin import translate as sw_to_text_translate

        translator, tokenizer = self._init_translator()

        # Prepare inputs with language prefix
        model_inputs = []
        for sw in signwriting_list:
            tokenized = " ".join(tokenizer.text_to_tokens(sw))
            model_inputs.append(f"${self.target_language} {tokenized}")

        logger.info(f"Translating {len(model_inputs)} SignWriting strings to text...")

        # Batch translate
        outputs = sw_to_text_translate(translator, model_inputs)

        # Clean up BPE tokens
        translations = [out.replace("@@", "") for out in outputs]

        logger.info(f"Translation complete: {translations}")
        return translations

    def process_video(self, video_path: str, video_id: str = None) -> dict:
        """
        Run the full pipeline on a single video.

        Args:
            video_path: Path to video file
            video_id: Optional video identifier

        Returns:
            Dictionary with results including timing information
        """
        timing = TimingResult()

        result = {
            "video_id": video_id or os.path.basename(video_path),
            "video_path": video_path,
            "signs_detected": 0,
            "signwriting": [],
            "translation": [],
            "full_text": "",
            "duration": 0.0,
            "fps": 0.0,
            "time_total_ms": 0.0,
            "time_pose_extraction_ms": 0.0,
            "time_segmentation_ms": 0.0,
            "time_transcription_ms": 0.0,
            "time_translation_ms": 0.0,
            "status": "success",
            "error": None
        }

        # Start total timer
        total_timer = CUDATimer(self.device)
        total_timer.start()

        try:
            # Step 1: Extract poses from video
            with TimedSection(CUDATimer(self.device)) as t:
                frames_data, width, height, fps = self.extract_poses_from_video(video_path)
            timing.pose_extraction_ms = t.elapsed_ms

            result["fps"] = fps
            result["duration"] = len(frames_data) / fps

            if len(frames_data) < 10:
                raise ValueError(f"Too few frames ({len(frames_data)}). Need at least 10.")

            # Step 2: Convert to Pose object (included in pose extraction time)
            pose = self.frames_to_pose(frames_data, width, height, fps)

            # Step 3: Run segmentation
            with TimedSection(CUDATimer(self.device)) as t:
                sign_segments, _ = self.segment_pose(pose)
            timing.segmentation_ms = t.elapsed_ms

            result["signs_detected"] = len(sign_segments)

            if not sign_segments:
                logger.warning("No signs detected in video")
                timing.total_ms = total_timer.stop()
                self._update_result_timing(result, timing)
                return result

            # Step 4: Transcribe to SignWriting
            with TimedSection(CUDATimer(self.device)) as t:
                signwriting_list = self.transcribe_to_signwriting(pose, sign_segments, fps)
            timing.transcription_ms = t.elapsed_ms

            result["signwriting"] = signwriting_list

            # Step 5: Translate to text
            with TimedSection(CUDATimer(self.device)) as t:
                translations = self.translate_signwriting_to_text(signwriting_list)
            timing.translation_ms = t.elapsed_ms

            result["translation"] = translations
            result["full_text"] = " ".join(translations)

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        timing.total_ms = total_timer.stop()
        self._update_result_timing(result, timing)

        logger.info(
            f"Timing for {video_id or video_path}: "
            f"total={timing.total_ms:.1f}ms, pose={timing.pose_extraction_ms:.1f}ms, "
            f"seg={timing.segmentation_ms:.1f}ms, transcribe={timing.transcription_ms:.1f}ms, "
            f"translate={timing.translation_ms:.1f}ms"
        )

        return result

    def _update_result_timing(self, result: dict, timing: TimingResult):
        """Update result dict with timing information."""
        result["time_total_ms"] = round(timing.total_ms, 2)
        result["time_pose_extraction_ms"] = round(timing.pose_extraction_ms, 2)
        result["time_segmentation_ms"] = round(timing.segmentation_ms, 2)
        result["time_transcription_ms"] = round(timing.transcription_ms, 2)
        result["time_translation_ms"] = round(timing.translation_ms, 2)

    def process_csv(
        self,
        input_csv: str,
        output_csv: str,
        video_column: str = "video_id",
        video_base_path: str = "/work3/s235253/openaslcropped",
        video_extension: str = ".mp4"
    ) -> list[dict]:
        """
        Process all videos from a CSV file.

        Args:
            input_csv: Path to input CSV with video IDs
            output_csv: Path to output CSV for results
            video_column: Name of column containing video IDs (default: video_id)
            video_base_path: Base directory where videos are located
            video_extension: Video file extension (default: .mp4)

        Returns:
            List of result dictionaries
        """
        # Read input CSV
        with open(input_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"No rows found in {input_csv}")

        if video_column not in rows[0]:
            # Try to find video column
            possible_cols = ['video_id', 'video_path', 'path', 'video', 'file', 'filename', 'id']
            for col in possible_cols:
                if col in rows[0]:
                    video_column = col
                    break
            else:
                raise ValueError(
                    f"Column '{video_column}' not found. "
                    f"Available columns: {list(rows[0].keys())}"
                )

        logger.info(f"Processing {len(rows)} videos from {input_csv}")
        logger.info(f"Video base path: {video_base_path}")
        logger.info(f"Video extension: {video_extension}")

        results = []

        for row in tqdm(rows, desc="Processing videos"):
            video_id = row[video_column]

            # Construct full video path: base_path/video_id.mp4
            video_path = os.path.join(video_base_path, f"{video_id}{video_extension}")

            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                results.append({
                    "video_id": video_id,
                    "video_path": video_path,
                    "signs_detected": 0,
                    "signwriting": [],
                    "translation": [],
                    "full_text": "",
                    "duration": 0.0,
                    "fps": 0.0,
                    "time_total_ms": 0.0,
                    "time_pose_extraction_ms": 0.0,
                    "time_segmentation_ms": 0.0,
                    "time_transcription_ms": 0.0,
                    "time_translation_ms": 0.0,
                    "status": "error",
                    "error": "File not found"
                })
                continue

            result = self.process_video(video_path, video_id=video_id)
            results.append(result)

        # Write output CSV
        fieldnames = [
            "video_id", "video_path", "signs_detected", "signwriting", "translation",
            "full_text", "duration", "fps",
            "time_total_ms", "time_pose_extraction_ms", "time_segmentation_ms",
            "time_transcription_ms", "time_translation_ms",
            "status", "error"
        ]

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Convert lists to strings for CSV
                row = result.copy()
                row["signwriting"] = "|".join(row["signwriting"])
                row["translation"] = "|".join(row["translation"])
                writer.writerow(row)

        logger.info(f"Results written to {output_csv}")

        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count
        total_signs = sum(r["signs_detected"] for r in results)

        logger.info(f"Summary: {success_count} succeeded, {error_count} failed, {total_signs} total signs")

        return results

    def close(self):
        """Release resources."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for SignWriting translation pipeline"
    )
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Input CSV file with video IDs"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--video-column",
        default="video_id",
        help="Name of column containing video IDs (default: video_id)"
    )
    parser.add_argument(
        "--video-base-path",
        default="/work3/s235253",
        help="Base directory where videos are located (default: /work3/s235253)"
    )
    parser.add_argument(
        "--video-extension",
        default=".mp4",
        help="Video file extension (default: .mp4)"
    )
    parser.add_argument(
        "--experiment-dir",
        default="experiment",
        help="Directory for model files (default: experiment)"
    )
    parser.add_argument(
        "--target-language",
        default="en",
        help="Target language code (default: en)"
    )
    parser.add_argument(
        "--single-video",
        help="Process a single video file path instead of CSV"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = SignWritingBatchInference(
        experiment_dir=args.experiment_dir,
        target_language=args.target_language,
        device=args.device
    )

    try:
        if args.single_video:
            # Process single video
            result = pipeline.process_video(args.single_video)
            print(f"\nResult for {args.single_video}:")
            print(f"  Signs detected: {result['signs_detected']}")
            print(f"  SignWriting: {result['signwriting']}")
            print(f"  Translation: {result['full_text']}")
            print(f"  Status: {result['status']}")
            print(f"  Timing (ms): total={result['time_total_ms']:.1f}, "
                  f"pose={result['time_pose_extraction_ms']:.1f}, "
                  f"seg={result['time_segmentation_ms']:.1f}, "
                  f"transcribe={result['time_transcription_ms']:.1f}, "
                  f"translate={result['time_translation_ms']:.1f}")
            if result['error']:
                print(f"  Error: {result['error']}")
        else:
            # Process CSV
            pipeline.process_csv(
                args.csv,
                args.output,
                video_column=args.video_column,
                video_base_path=args.video_base_path,
                video_extension=args.video_extension
            )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()