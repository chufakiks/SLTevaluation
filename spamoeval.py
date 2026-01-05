#!/usr/bin/env python3
"""Evaluate SPAMO translations against OpenASL references using BLEU.

Uses SacreBLEU with settings matching SpaMo paper:
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sacrebleu.metrics import BLEU


def evaluate_results(predictions, references, tokenizer='13a'):
    """
    Evaluate prediction results using BLEU metrics (matching SpaMo paper settings).

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (str): Tokenizer for BLEU scoring (default: 13a).

    Returns:
        dict: A dictionary of BLEU-1 through BLEU-4 scores.
    """
    scores = {}

    for i in range(1, 5):
        bleu = BLEU(max_ngram_order=i, tokenize=tokenizer, smooth_method='exp')
        score = bleu.corpus_score(predictions, [references]).score
        scores[f"BLEU-{i}"] = score

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate SPAMO translations")
    parser.add_argument("--results", type=str, default="spamores/results_en.csv",
                        help="Path to results CSV with translation_en column")
    parser.add_argument("--references", type=str, default="openasl-v1.0.tsv",
                        help="Path to OpenASL TSV file with reference translations")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save evaluation results")
    args = parser.parse_args()

    # Load results (predictions)
    print(f"Loading results from {args.results}...")
    results_df = pd.read_csv(args.results)

    # Load references
    print(f"Loading references from {args.references}...")
    refs_df = pd.read_csv(args.references, sep='\t')

    # Create lookup dict from vid to raw-text
    ref_lookup = dict(zip(refs_df['vid'], refs_df['raw-text']))

    # Match predictions with references
    predictions = []
    references = []
    matched = 0
    unmatched = 0

    for _, row in results_df.iterrows():
        video_id = row['video_id']
        if video_id in ref_lookup:
            pred = row.get('translation_en', row.get('translation', ''))
            ref = ref_lookup[video_id]
            if pd.notna(pred) and pd.notna(ref):
                predictions.append(str(pred))
                references.append(str(ref))
                matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")

    if matched == 0:
        print("Error: No matching video IDs found between results and references.")
        return

    # Evaluate
    print("Evaluating with SacreBLEU (tok:13a, smooth:exp)...")
    scores = evaluate_results(predictions, references)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in scores.items():
        print(f"{metric}: {value:.2f}")

    # Print average timing per frame if timing columns exist
    timing_cols = ['frame_extraction_time', 'spatial_feature_time', 'motion_feature_time', 'inference_time', 'total_time']
    if all(col in results_df.columns for col in timing_cols) and 'num_frames' in results_df.columns:
        total_frames = results_df['num_frames'].sum()
        print("\n" + "="*50)
        print("AVERAGE TIMING PER FRAME")
        print("="*50)
        print(f"Total frames processed:      {total_frames}")
        print(f"Frame extraction:            {results_df['frame_extraction_time'].sum() / total_frames * 1000:.3f}ms")
        print(f"Spatial feature extraction:  {results_df['spatial_feature_time'].sum() / total_frames * 1000:.3f}ms")
        print(f"Motion feature extraction:   {results_df['motion_feature_time'].sum() / total_frames * 1000:.3f}ms")
        print(f"Model inference:             {results_df['inference_time'].sum() / total_frames * 1000:.3f}ms")
        print(f"Total time per frame:        {results_df['total_time'].sum() / total_frames * 1000:.3f}ms")

        # Generate scatter plot: frame length vs visual feature extraction time
        visual_time = results_df['spatial_feature_time'] + results_df['motion_feature_time']
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['num_frames'], visual_time, alpha=0.5, s=10)
        plt.xlabel('Number of Frames')
        plt.ylabel('Visual Feature Extraction Time (s)')
        plt.title('Frame Length vs Visual Feature Extraction Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = args.output.replace('.csv', '_scatter.png') if args.output else 'timing_scatter.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nScatter plot saved to {plot_path}")

    # Save results if output specified
    if args.output:
        output_data = scores.copy()
        # Add per-frame timing (in ms) if available
        if all(col in results_df.columns for col in timing_cols) and 'num_frames' in results_df.columns:
            total_frames = results_df['num_frames'].sum()
            output_data['total_frames'] = total_frames
            output_data['per_frame_extraction_ms'] = results_df['frame_extraction_time'].sum() / total_frames * 1000
            output_data['per_frame_spatial_ms'] = results_df['spatial_feature_time'].sum() / total_frames * 1000
            output_data['per_frame_motion_ms'] = results_df['motion_feature_time'].sum() / total_frames * 1000
            output_data['per_frame_inference_ms'] = results_df['inference_time'].sum() / total_frames * 1000
            output_data['per_frame_total_ms'] = results_df['total_time'].sum() / total_frames * 1000
        scores_df = pd.DataFrame([output_data])
        scores_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()