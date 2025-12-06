#!/usr/bin/env python3
"""
Analyze LoCoDiff evaluation results and generate plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_results(json_file: Path) -> List[Dict]:
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_similarities(results: List[Dict]) -> Dict[str, float]:
    """Extract similarity scores by example ID."""
    return {r['id']: r['similarity'] for r in results if r.get('success', False)}


def bucket_context(tokens: int) -> str:
    """Bucket examples by context length (prompt tokens)."""
    if tokens < 10_000:
        return "Small (<10k)"
    if tokens < 25_000:
        return "Medium (10-25k)"
    return "Large (25k+)"


def analyze_results(baseline_file: Path, rlm_file: Path, summary_file: Path):
    """Analyze and plot results."""
    
    # Load data
    baseline_results = load_results(baseline_file)
    rlm_results = load_results(rlm_file)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Extract similarities
    baseline_sims = extract_similarities(baseline_results)
    rlm_sims = extract_similarities(rlm_results)
    
    # Get common examples
    common_ids = sorted(set(baseline_sims.keys()) & set(rlm_sims.keys()))
    
    baseline_values = [baseline_sims[id] for id in common_ids]
    rlm_values = [rlm_sims[id] for id in common_ids]
    
    # Calculate statistics
    baseline_avg = np.mean(baseline_values)
    rlm_avg = np.mean(rlm_values)
    baseline_std = np.std(baseline_values)
    rlm_std = np.std(rlm_values)
    
    # Calculate improvement
    improvements = [(r - b) * 100 for r, b in zip(rlm_values, baseline_values)]
    avg_improvement = np.mean(improvements)

    # Prompt tokens (proxy for context length) and bucketing
    baseline_tokens = {r['id']: r.get('prompt_tokens', 0) for r in baseline_results if r.get('success', False)}
    rlm_tokens = {r['id']: r.get('prompt_tokens', 0) for r in rlm_results if r.get('success', False)}

    buckets = ["Small (<10k)", "Medium (10-25k)", "Large (25k+)"]
    bucket_colors = {
        "Small (<10k)": "#4CAF50",
        "Medium (10-25k)": "#2196F3",
        "Large (25k+)": "#FF9800",
    }
    per_bucket = {b: {"baseline": [], "rlm": [], "improvements": [], "count": 0} for b in buckets}
    for idx, bid in enumerate(common_ids):
        tokens_val = baseline_tokens.get(bid, rlm_tokens.get(bid, 0))
        bucket = bucket_context(tokens_val)
        per_bucket[bucket]["baseline"].append(baseline_values[idx])
        per_bucket[bucket]["rlm"].append(rlm_values[idx])
        per_bucket[bucket]["improvements"].append(improvements[idx])
        per_bucket[bucket]["count"] += 1
    
    # Output directory for plots
    plots_dir = baseline_file.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Scatter plot: Baseline vs RLM
    fig, ax1 = plt.subplots(figsize=(6, 5))
    scatter_colors = [
        bucket_colors[bucket_context(baseline_tokens.get(id, rlm_tokens.get(id, 0)))]
        for id in common_ids
    ]
    ax1.scatter(baseline_values, rlm_values, alpha=0.6, s=60, c=scatter_colors)
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x (no improvement)')
    ax1.set_xlabel('Baseline Similarity', fontsize=11)
    ax1.set_ylabel('RLM Similarity', fontsize=11)
    ax1.set_title('Baseline vs RLM Similarity', fontsize=12, fontweight='bold')
    # Legend for buckets
    for name, color in bucket_colors.items():
        ax1.scatter([], [], c=color, label=name, s=60)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add diagonal improvement zones
    ax1.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green', label='RLM better')
    ax1.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red', label='Baseline better')
    
    fig.savefig(plots_dir / "evaluation_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Bar chart: Similarity by context bucket (no per-example labels)
    fig, ax2 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(buckets))
    width = 0.35
    bucket_baseline_means = [np.mean(per_bucket[b]["baseline"]) if per_bucket[b]["baseline"] else 0 for b in buckets]
    bucket_rlm_means = [np.mean(per_bucket[b]["rlm"]) if per_bucket[b]["rlm"] else 0 for b in buckets]
    ax2.bar(x - width/2, bucket_baseline_means, width, label='Baseline', alpha=0.8)
    ax2.bar(x + width/2, bucket_rlm_means, width, label='RLM', alpha=0.8)
    ax2.set_xlabel('Context bucket', fontsize=11)
    ax2.set_ylabel('Similarity', fontsize=11)
    ax2.set_title('Average Similarity by Context Size', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b}\n(n={per_bucket[b]['count']})" for b in buckets], rotation=0, fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    fig.savefig(plots_dir / "evaluation_similarity_by_bucket.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Improvement distribution (overall)
    fig, ax3 = plt.subplots(figsize=(6, 5))
    ax3.hist(improvements, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax3.axvline(avg_improvement, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {avg_improvement:.2f}%')
    ax3.set_xlabel('Improvement (%)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('RLM Improvement Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.savefig(plots_dir / "evaluation_improvement_hist.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. Box plot comparison by context bucket (baseline vs RLM)
    fig, ax4 = plt.subplots(figsize=(7, 5))
    data_to_plot = []
    labels = []
    colors_boxes = []
    for b in buckets:
        if per_bucket[b]["baseline"]:
            data_to_plot.append(per_bucket[b]["baseline"])
            labels.append(f"Baseline\n{b}")
            colors_boxes.append('#90CAF9')
        if per_bucket[b]["rlm"]:
            data_to_plot.append(per_bucket[b]["rlm"])
            labels.append(f"RLM\n{b}")
            colors_boxes.append('#A5D6A7')

    bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    for box, c in zip(bp['boxes'], colors_boxes):
        box.set_facecolor(c)
    ax4.set_ylabel('Similarity', fontsize=11)
    ax4.set_title('Similarity Distribution by Context Bucket', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)
    
    fig.savefig(plots_dir / "evaluation_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5. Summary statistics table
    fig, ax5 = plt.subplots(figsize=(6, 5))
    ax5.axis('off')
    stats_data = [
        ['Metric', 'Baseline', 'RLM', 'Difference'],
        ['Mean', f'{baseline_avg:.3f}', f'{rlm_avg:.3f}', f'{rlm_avg - baseline_avg:+.3f}'],
        ['Std Dev', f'{baseline_std:.3f}', f'{rlm_std:.3f}', f'{rlm_std - baseline_std:+.3f}'],
        ['Min', f'{min(baseline_values):.3f}', f'{min(rlm_values):.3f}', f'{min(rlm_values) - min(baseline_values):+.3f}'],
        ['Max', f'{max(baseline_values):.3f}', f'{max(rlm_values):.3f}', f'{max(rlm_values) - max(baseline_values):+.3f}'],
        ['Avg Time (s)', f"{summary['configurations']['baseline_none_none_parallel=False']['avg_time']:.1f}",
         f"{summary['configurations']['rlm_none_none_parallel=False']['avg_time']:.1f}", 
         f"{summary['configurations']['rlm_none_none_parallel=False']['avg_time'] - summary['configurations']['baseline_none_none_parallel=False']['avg_time']:+.1f}"],
        ['LLM Calls', f"{summary['configurations']['baseline_none_none_parallel=False']['avg_llm_calls']:.1f}",
         f"{summary['configurations']['rlm_none_none_parallel=False']['avg_llm_calls']:.1f}",
         f"{summary['configurations']['rlm_none_none_parallel=False']['avg_llm_calls'] - summary['configurations']['baseline_none_none_parallel=False']['avg_llm_calls']:+.1f}"],
        ['Buckets (n)', 
         ', '.join([f"{b.split()[0]}={per_bucket[b]['count']}" for b in buckets]),
         '', ''],
    ]
    table = ax5.table(cellText=stats_data[1:], colLabels=stats_data[0],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Style header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    fig.savefig(plots_dir / "evaluation_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 6. Win/Loss/Tie breakdown (overall)
    fig, ax6 = plt.subplots(figsize=(5.5, 5))
    wins = sum(1 for imp in improvements if imp > 0)
    losses = sum(1 for imp in improvements if imp < 0)
    ties = sum(1 for imp in improvements if imp == 0)
    categories = ['RLM Wins', 'Baseline Wins', 'Ties']
    counts = [wins, losses, ties]
    colors = ['green', 'red', 'gray']
    bars = ax6.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Win/Loss/Tie Breakdown', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(improvements)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    fig.savefig(plots_dir / "evaluation_win_loss_tie.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 7. Similarity by file size (approximate) with bucket color
    fig, ax7 = plt.subplots(figsize=(6.5, 5))
    # Use prompt tokens as proxy for file size (already computed baseline_tokens)
    tokens = [baseline_tokens.get(id, 0) for id in common_ids]
    
    ax7.scatter(tokens, baseline_values, alpha=0.6, label='Baseline', s=60, marker='o', c='gray')
    ax7.scatter(tokens, rlm_values, alpha=0.6, label='RLM', s=60, marker='^', c=[bucket_colors[bucket_context(baseline_tokens.get(id, 0))] for id in common_ids])
    ax7.set_xlabel('Prompt Tokens (proxy for file size)', fontsize=11)
    ax7.set_ylabel('Similarity', fontsize=11)
    ax7.set_title('Similarity vs File Size', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 1)
    
    fig.savefig(plots_dir / "evaluation_tokens_vs_similarity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 8. Time vs Similarity tradeoff
    fig, ax8 = plt.subplots(figsize=(6.5, 5))
    baseline_times = {r['id']: r.get('elapsed_time', 0) for r in baseline_results if r.get('success', False)}
    rlm_times = {r['id']: r.get('elapsed_time', 0) for r in rlm_results if r.get('success', False)}
    baseline_time_vals = [baseline_times.get(id, 0) for id in common_ids]
    rlm_time_vals = [rlm_times.get(id, 0) for id in common_ids]

    ax8.scatter(baseline_time_vals, baseline_values, alpha=0.6, label='Baseline', s=60)
    ax8.scatter(rlm_time_vals, rlm_values, alpha=0.6, label='RLM', s=60, marker='^')
    ax8.set_xlabel('Time (seconds)', fontsize=11)
    ax8.set_ylabel('Similarity', fontsize=11)
    ax8.set_title('Time vs Similarity Tradeoff', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    
    fig.savefig(plots_dir / "evaluation_time_vs_similarity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Saved plots to", plots_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total examples: {len(common_ids)}")
    print(f"\nBaseline:")
    print(f"  Mean similarity: {baseline_avg:.3f} ± {baseline_std:.3f}")
    print(f"  Range: [{min(baseline_values):.3f}, {max(baseline_values):.3f}]")
    print(f"\nRLM:")
    print(f"  Mean similarity: {rlm_avg:.3f} ± {rlm_std:.3f}")
    print(f"  Range: [{min(rlm_values):.3f}, {max(rlm_values):.3f}]")
    print(f"\nImprovement:")
    print(f"  Average: {avg_improvement:+.2f} percentage points")
    print(f"  RLM wins: {wins} ({wins/len(improvements)*100:.1f}%)")
    print(f"  Baseline wins: {losses} ({losses/len(improvements)*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/len(improvements)*100:.1f}%)")
    print(f"\nPerformance:")
    print(f"  Baseline avg time: {summary['configurations']['baseline_none_none_parallel=False']['avg_time']:.1f}s")
    print(f"  RLM avg time: {summary['configurations']['rlm_none_none_parallel=False']['avg_time']:.1f}s")
    print(f"  Time overhead: {(summary['configurations']['rlm_none_none_parallel=False']['avg_time'] / summary['configurations']['baseline_none_none_parallel=False']['avg_time'] - 1) * 100:.1f}%")
    print(f"\nBy context bucket (n per bucket):")
    for b in buckets:
        print(f"  {b}: {per_bucket[b]['count']} examples")
    print("="*60)


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "locodiff_results"
    
    # Find the most recent files
    baseline_files = list(results_dir.glob("baseline_*.json"))
    rlm_files = list(results_dir.glob("rlm_none_unfiltered_parallel=False_*.json"))
    summary_files = list(results_dir.glob("summary_*.json"))
    
    if not baseline_files or not rlm_files or not summary_files:
        print("❌ Error: Could not find result files!")
        print(f"  Looking in: {results_dir}")
        exit(1)
    
    # Get most recent
    baseline_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    rlm_file = max(rlm_files, key=lambda p: p.stat().st_mtime)
    summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Using files:")
    print(f"  Baseline: {baseline_file.name}")
    print(f"  RLM: {rlm_file.name}")
    print(f"  Summary: {summary_file.name}")
    print()
    
    analyze_results(baseline_file, rlm_file, summary_file)

