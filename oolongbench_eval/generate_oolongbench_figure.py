"""
Generate figures for OOLONGBench evaluation results.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def generate_figures(results_dir="oolongbench_results"):
    """Generate performance comparison figures."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found.")
        return

    # Load all results
    all_results = []
    for file in results_path.glob("*.json"):
        if file.name.startswith("summary"):
            continue
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not all_results:
        print("No results found to plot.")
        return

    # Group results by configuration
    stats = defaultdict(lambda: {'f1': [], 'em': [], 'time': [], 'calls': []})
    
    for r in all_results:
        if not r.get('success'):
            continue
            
        # Create a readable label for the configuration
        strategy = r.get('partition_strategy', 'unknown')
        retrieval = r.get('retrieval_method', 'unknown')
        label = f"{strategy} + {retrieval}"
        
        stats[label]['f1'].append(r.get('f1_score', 0))
        stats[label]['em'].append(1.0 if r.get('exact_match') else 0.0)
        stats[label]['time'].append(r.get('elapsed_time', 0))
        stats[label]['calls'].append(r.get('llm_calls', 0))

    # Calculate averages
    labels = []
    avg_f1 = []
    avg_em = []
    avg_time = []
    
    for label, metrics in sorted(stats.items()):
        labels.append(label)
        avg_f1.append(np.mean(metrics['f1']))
        avg_em.append(np.mean(metrics['em']))
        avg_time.append(np.mean(metrics['time']))

    # Plot F1 and Exact Match
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, avg_f1, width, label='F1 Score')
    rects2 = ax.bar(x + width/2, avg_em, width, label='Exact Match')

    ax.set_ylabel('Score')
    ax.set_title('OOLONGBench Performance by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    plt.tight_layout()
    plt.savefig(results_path / 'performance_comparison.png')
    print(f"Saved performance figure to {results_path / 'performance_comparison.png'}")

    # Plot Latency
    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_time, color='orange')
    plt.ylabel('Average Time (s)')
    plt.title('Average Latency by Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_path / 'latency_comparison.png')
    print(f"Saved latency figure to {results_path / 'latency_comparison.png'}")

if __name__ == "__main__":
    generate_figures()

