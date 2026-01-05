#!/usr/bin/env python3
"""Evaluate SPAMO translations against OpenASL references using BLEU.

Uses SacreBLEU with settings matching SpaMo paper:
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
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

        # Generate scatter plots
        base_path = args.output.replace('.csv', '') if args.output else 'timing'

        # Filter out rows with NaN values for regression
        valid_mask = results_df[['num_frames', 'spatial_feature_time', 'motion_feature_time']].notna().all(axis=1)
        valid_df = results_df[valid_mask]

        # Linear regression for spatial and motion
        x = valid_df['num_frames'].values
        y_spatial = valid_df['spatial_feature_time'].values
        y_motion = valid_df['motion_feature_time'].values
        slope_s, intercept_s, r_s, _, _ = stats.linregress(x, y_spatial)
        slope_m, intercept_m, r_m, _, _ = stats.linregress(x, y_motion)

        # Generate line points for plotting (min to max)
        x_min, x_max = x.min(), x.max()
        x_line = [x_min, x_max]

        # 1. Frame length vs spatial feature extraction time (with linear regression)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y_spatial, alpha=0.5, s=10)
        plt.plot(x_line, [slope_s * xi + intercept_s for xi in x_line], 'r-', linewidth=2)
        plt.xlabel('Number of Frames')
        plt.ylabel('Spatial Feature Extraction Time (s)')
        plt.title('Frame Length vs Spatial Feature Extraction Time')
        # Add regression equation as text annotation
        plt.text(0.05, 0.95, f'y = {slope_s*1000:.2f}ms/frame + {intercept_s*1000:.2f}ms\nR² = {r_s**2:.4f}',
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_spatial.png", dpi=150)
        plt.close()

        # 2. Frame length vs motion feature extraction time (with linear regression)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y_motion, alpha=0.5, s=10)
        plt.plot(x_line, [slope_m * xi + intercept_m for xi in x_line], 'r-', linewidth=2)
        plt.xlabel('Number of Frames')
        plt.ylabel('Motion Feature Extraction Time (s)')
        plt.title('Frame Length vs Motion Feature Extraction Time')
        # Add regression equation as text annotation
        plt.text(0.05, 0.95, f'y = {slope_m*1000:.2f}ms/frame + {intercept_m*1000:.2f}ms\nR² = {r_m**2:.4f}',
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_motion.png", dpi=150)
        plt.close()

        # 3. Frame length vs inference time (with logarithmic fit)
        y_inference = valid_df['inference_time'].values

        # Logarithmic fit: y = a * log(x) + b
        def log_func(x, a, b):
            return a * np.log(x) + b

        try:
            popt, _ = curve_fit(log_func, x, y_inference)
            a_inf, b_inf = popt
            # Calculate R² for log fit
            y_pred = log_func(x, a_inf, b_inf)
            ss_res = np.sum((y_inference - y_pred) ** 2)
            ss_tot = np.sum((y_inference - np.mean(y_inference)) ** 2)
            r2_inf = 1 - (ss_res / ss_tot)
            log_fit_success = True
        except Exception:
            log_fit_success = False

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y_inference, alpha=0.5, s=10)
        if log_fit_success:
            x_smooth = np.linspace(x_min, x_max, 100)
            plt.plot(x_smooth, log_func(x_smooth, a_inf, b_inf), 'r-', linewidth=2)
            plt.text(0.05, 0.95, f'y = {a_inf*1000:.2f}ms·log(x) + {b_inf*1000:.2f}ms\nR² = {r2_inf:.4f}',
                     transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('Number of Frames')
        plt.ylabel('Inference Time (s)')
        plt.title('Frame Length vs Inference Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_path}_inference.png", dpi=150)
        plt.close()

        print(f"\nScatter plots saved to {base_path}_spatial.png, {base_path}_motion.png, {base_path}_inference.png")
        print(f"\nLinear regression results:")
        print(f"  Spatial: {slope_s*1000:.3f}ms/frame + {intercept_s*1000:.3f}ms (R² = {r_s**2:.4f})")
        print(f"  Motion:  {slope_m*1000:.3f}ms/frame + {intercept_m*1000:.3f}ms (R² = {r_m**2:.4f})")
        if log_fit_success:
            print(f"  Inference (log): {a_inf*1000:.3f}ms·log(frames) + {b_inf*1000:.3f}ms (R² = {r2_inf:.4f})")

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