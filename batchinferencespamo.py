"""
Batch inference script for SpaMo sign language translation.
Optimized for GPU with CUDA - efficient memory usage and parallel processing.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gc

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    CLIPVisionModel,
    VideoMAEImageProcessor,
    VideoMAEModel,
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import instantiate_from_config, sliding_window_for_list
from utils.s2wrapper import forward as multiscale_forward


@dataclass
class InferenceConfig:
    """Configuration for batch inference."""
    checkpoint: str
    config: str = 'configs/finetune.yaml'
    device: str = 'cuda:0'
    target_lang: str = 'German'
    spatial_batch_size: int = 64  # Larger batch for GPU
    motion_batch_size: int = 16   # Larger batch for GPU
    max_frames: Optional[int] = None
    cache_dir: Optional[str] = None
    save_features: bool = False
    use_fp16: bool = True  # Use mixed precision for faster inference


class CUDAMemoryManager:
    """Manage CUDA memory efficiently."""

    @staticmethod
    def clear_cache():
        """Clear CUDA cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    @staticmethod
    def get_memory_info(device: str = 'cuda:0') -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'total': 0}

        device_idx = int(device.split(':')[1]) if ':' in device else 0
        allocated = torch.cuda.memory_allocated(device_idx) / 1e9
        reserved = torch.cuda.memory_reserved(device_idx) / 1e9
        total = torch.cuda.get_device_properties(device_idx).total_memory / 1e9

        return {'allocated': allocated, 'reserved': reserved, 'total': total}

    @staticmethod
    def print_memory_usage(device: str = 'cuda:0', prefix: str = ''):
        """Print current GPU memory usage."""
        info = CUDAMemoryManager.get_memory_info(device)
        print(f"{prefix}GPU Memory: {info['allocated']:.2f}GB / {info['total']:.2f}GB")


class VideoFrameExtractor:
    """Extract frames from video files efficiently."""

    def extract_frames(self, video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
        """Extract frames from video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

                if max_frames and len(frames) >= max_frames:
                    break

            return frames
        finally:
            cap.release()


class GPUSpatialFeatureExtractor:
    """Extract spatial features using CLIP ViT with S2-Wrapper - GPU optimized."""

    def __init__(
        self,
        model_name: str = 'openai/clip-vit-large-patch14',
        scales: List[int] = [1, 2],
        device: str = 'cuda:0',
        cache_dir: Optional[str] = None,
        use_fp16: bool = True
    ):
        self.device = device
        self.scales = scales
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"Loading CLIP ViT model: {model_name}")
        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=cache_dir,
            torch_dtype=self.dtype
        ).to(device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Enable memory efficient attention if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        print(f"  Spatial extractor ready (FP16: {self.use_fp16})")

    @torch.no_grad()
    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLIP ViT."""
        outputs = self.model(inputs).hidden_states
        return outputs[-1]

    @torch.no_grad()
    def extract(self, frames: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Extract spatial features with GPU optimization."""
        all_features = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:min(i + batch_size, len(frames))]

            inputs = self.image_processor(
                list(batch_frames),
                return_tensors="pt"
            ).to(self.device)

            if self.use_fp16:
                inputs.pixel_values = inputs.pixel_values.half()

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                outputs = multiscale_forward(
                    self.forward_features,
                    inputs.pixel_values,
                    scales=self.scales,
                    num_prefix_token=1
                )

            features = outputs[:, 0].float().cpu().numpy()
            all_features.append(features)

        return np.concatenate(all_features, axis=0)


class GPUMotionFeatureExtractor:
    """Extract motion features using VideoMAE - GPU optimized."""

    def __init__(
        self,
        model_name: str = 'MCG-NJU/videomae-large',
        window_size: int = 16,
        overlap_size: int = 8,
        device: str = 'cuda:0',
        cache_dir: Optional[str] = None,
        use_fp16: bool = True
    ):
        self.device = device
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        print(f"Loading VideoMAE model: {model_name}")
        self.model = VideoMAEModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=self.dtype
        ).to(device).eval()

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        print(f"  Motion extractor ready (FP16: {self.use_fp16})")

    @torch.no_grad()
    def extract(self, frames: List[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Extract motion features with GPU optimization."""
        # Handle short videos
        if len(frames) < self.window_size:
            frames = frames + [frames[-1]] * (self.window_size - len(frames))

        frame_windows = sliding_window_for_list(
            frames,
            window_size=self.window_size,
            overlap_size=self.overlap_size
        )

        all_features = []

        for i in range(0, len(frame_windows), batch_size):
            batch_windows = frame_windows[i:min(i + batch_size, len(frame_windows))]

            inputs = self.image_processor(
                images=batch_windows,
                return_tensors="pt"
            ).to(self.device)

            if self.use_fp16:
                inputs.pixel_values = inputs.pixel_values.half()

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                outputs = self.model(**inputs, output_hidden_states=True).hidden_states
                features = outputs[-1][:, 0]

            all_features.append(features.float().cpu().numpy())

        return np.concatenate(all_features, axis=0)


class SpaMoBatchInference:
    """SpaMo model batch inference - GPU optimized."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda:0',
        use_fp16: bool = True
    ):
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        print(f"\nLoading SpaMo model...")
        self.config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(self.config.model)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            # Partial load
            model_state = self.model.state_dict()
            ckpt_state = checkpoint['state_dict']
            filtered = {k: v for k, v in ckpt_state.items()
                       if k in model_state and v.size() == model_state[k].size()}
            self.model.load_state_dict(filtered, strict=False)
            print(f"  Loaded {len(filtered)}/{len(ckpt_state)} parameters")

        self.model = self.model.to(device).eval()

        if self.use_fp16:
            # Convert model components that support FP16
            self.model.half()

        print(f"  Model ready (FP16: {self.use_fp16})")

    @torch.no_grad()
    def infer(
        self,
        spatial_features: np.ndarray,
        motion_features: np.ndarray,
        target_lang: str = 'German',
        video_id: str = 'video'
    ) -> str:
        """Run inference on extracted features."""
        dtype = torch.float16 if self.use_fp16 else torch.float32

        spatial_feat = torch.tensor(spatial_features, dtype=dtype)
        motion_feat = torch.tensor(motion_features, dtype=dtype)

        sample = {
            'pixel_values': [spatial_feat],
            'glor_values': [motion_feat],
            'num_frames': [len(spatial_feat)],
            'glor_lengths': [len(motion_feat)],
            'lang': [target_lang],
            'text': [''],
            'ex_lang_trans': [''],
            'ids': [video_id]
        }

        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            visual_outputs, visual_masks = self.model.prepare_visual_inputs(sample)
            visual_outputs = self.model.fusion_proj(visual_outputs)

            input_embeds, input_masks, _, _ = self.model.prepare_inputs(
                visual_outputs,
                visual_masks,
                sample,
                split='test',
                batch_idx=0
            )

            generated = self.model.t5_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=input_masks,
                num_beams=5,
                max_length=64,
                top_p=0.9,
                do_sample=True,
            )

        translation = self.model.t5_tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )[0].lower()

        return translation


class BatchInferencePipeline:
    """Complete pipeline for batch inference on multiple videos."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device

        # Check CUDA availability
        if 'cuda' in self.device and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'
            config.use_fp16 = False

        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            CUDAMemoryManager.print_memory_usage(self.device, "Initial ")

        # Initialize components
        self.frame_extractor = VideoFrameExtractor()

        print("\n--- Loading Feature Extractors ---")
        self.spatial_extractor = GPUSpatialFeatureExtractor(
            device=self.device,
            cache_dir=config.cache_dir,
            use_fp16=config.use_fp16
        )

        self.motion_extractor = GPUMotionFeatureExtractor(
            device=self.device,
            cache_dir=config.cache_dir,
            use_fp16=config.use_fp16
        )

        print("\n--- Loading SpaMo Model ---")
        self.model = SpaMoBatchInference(
            checkpoint_path=config.checkpoint,
            config_path=config.config,
            device=self.device,
            use_fp16=config.use_fp16
        )

        if torch.cuda.is_available():
            CUDAMemoryManager.print_memory_usage(self.device, "After loading ")

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """Process a single video and return results."""
        video_id = Path(video_path).stem

        # Extract frames
        frames = self.frame_extractor.extract_frames(
            video_path,
            self.config.max_frames
        )

        if len(frames) == 0:
            return {'video_id': video_id, 'error': 'No frames extracted'}

        # Extract features
        spatial_features = self.spatial_extractor.extract(
            frames,
            batch_size=self.config.spatial_batch_size
        )

        motion_features = self.motion_extractor.extract(
            frames,
            batch_size=self.config.motion_batch_size
        )

        # Save features if requested
        if self.config.save_features and output_dir:
            np.save(output_dir / f"{video_id}_s2wrapping.npy", spatial_features)
            np.save(output_dir / f"{video_id}_overlap-8-pretrained.npy", motion_features)

        # Run inference
        translation = self.model.infer(
            spatial_features=spatial_features,
            motion_features=motion_features,
            target_lang=self.config.target_lang,
            video_id=video_id
        )

        # Clear some memory
        del frames
        CUDAMemoryManager.clear_cache()

        return {
            'video_id': video_id,
            'video_path': video_path,
            'translation': translation,
            'num_frames': len(spatial_features),
            'spatial_shape': spatial_features.shape,
            'motion_shape': motion_features.shape
        }

    def process_batch(
        self,
        video_paths: List[str],
        output_dir: Path
    ) -> List[Dict]:
        """Process multiple videos with progress tracking."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        print(f"\nProcessing {len(video_paths)} videos...")
        print("=" * 60)

        for video_path in tqdm(video_paths, desc="Processing videos"):
            try:
                result = self.process_video(video_path, output_dir)
                results.append(result)

                if 'error' not in result:
                    tqdm.write(f"  {result['video_id']}: {result['translation'][:50]}...")
                else:
                    tqdm.write(f"  {result['video_id']}: ERROR - {result['error']}")

            except Exception as e:
                results.append({
                    'video_id': Path(video_path).stem,
                    'video_path': video_path,
                    'error': str(e)
                })
                tqdm.write(f"  {Path(video_path).stem}: ERROR - {e}")

        return results


def find_videos(directory: str, patterns: List[str] = ['*.mp4', '*.avi', '*.mov']) -> List[str]:
    """Find all video files in directory matching patterns."""
    video_paths = []
    directory = Path(directory)

    for pattern in patterns:
        video_paths.extend(directory.glob(pattern))
        video_paths.extend(directory.glob(f"**/{pattern}"))  # Recursive

    return sorted(set(str(p) for p in video_paths))


def save_results(results: List[Dict], output_dir: Path, format: str = 'all'):
    """Save results to files."""
    # Save as JSON
    if format in ['all', 'json']:
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {json_path}")

    # Save as CSV
    if format in ['all', 'csv']:
        csv_path = output_dir / 'results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'translation', 'num_frames', 'error'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'video_id': r.get('video_id', ''),
                    'translation': r.get('translation', ''),
                    'num_frames': r.get('num_frames', ''),
                    'error': r.get('error', '')
                })
        print(f"Results saved to: {csv_path}")

    # Save translations as text
    if format in ['all', 'txt']:
        txt_path = output_dir / 'translations.txt'
        with open(txt_path, 'w') as f:
            for r in results:
                if 'translation' in r:
                    f.write(f"{r['video_id']}\t{r['translation']}\n")
        print(f"Translations saved to: {txt_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch inference for SpaMo - GPU optimized'
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video_dir',
        type=str,
        help='Directory containing video files'
    )
    input_group.add_argument(
        '--video_list',
        type=str,
        help='Text file with video paths (one per line)'
    )
    input_group.add_argument(
        '--video_path',
        type=str,
        help='Single video file path'
    )

    # Required
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Optional
    parser.add_argument('--config', type=str, default='configs/finetune.yaml')
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--target_lang', type=str, default='German',
                       choices=['German', 'English', 'French', 'Spanish', 'Chinese'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--spatial_batch_size', type=int, default=64)
    parser.add_argument('--motion_batch_size', type=int, default=16)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--save_features', action='store_true')
    parser.add_argument('--no_fp16', action='store_true', help='Disable FP16 inference')
    parser.add_argument('--output_format', type=str, default='all',
                       choices=['all', 'json', 'csv', 'txt'])

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("SPAMO BATCH INFERENCE - GPU OPTIMIZED")
    print("=" * 60)

    # Collect video paths
    if args.video_path:
        video_paths = [args.video_path]
    elif args.video_list:
        with open(args.video_list) as f:
            video_paths = [line.strip() for line in f if line.strip()]
    else:
        video_paths = find_videos(args.video_dir)

    print(f"Found {len(video_paths)} video(s)")

    if not video_paths:
        print("No videos found!")
        return

    # Create config
    config = InferenceConfig(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        target_lang=args.target_lang,
        spatial_batch_size=args.spatial_batch_size,
        motion_batch_size=args.motion_batch_size,
        max_frames=args.max_frames,
        cache_dir=args.cache_dir,
        save_features=args.save_features,
        use_fp16=not args.no_fp16
    )

    # Initialize pipeline
    pipeline = BatchInferencePipeline(config)

    # Process videos
    output_dir = Path(args.output_dir)
    results = pipeline.process_batch(video_paths, output_dir)

    # Save results
    print("\n" + "=" * 60)
    save_results(results, output_dir, args.output_format)

    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    print(f"\nCompleted: {successful}/{len(results)} videos processed successfully")

    if torch.cuda.is_available():
        CUDAMemoryManager.print_memory_usage(args.device, "Final ")


if __name__ == '__main__':
    main()
