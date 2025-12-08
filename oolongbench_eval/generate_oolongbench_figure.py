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
    stats = defaultdict(lambda: {'oolong': [], 'f1': [], 'em': [], 'time': [], 'calls': []})
    
    for r in all_results:
        if not r.get('success'):
            continue
            
        # Determine configuration label
        mode = r.get('mode', 'rlm')
        if mode == 'direct':
            label = "GPT-5-mini (Direct)"
        elif mode == 'rlm-no-recursion':
            label = "RLM (No Recursion)"
        else:
            # Check for REPL default (no partition strategy)
            strategy = r.get('partition_strategy')
            retrieval = r.get('retrieval_method')
            
            if not strategy:
                label = "RLM (REPL Default)"
            else:
                label = f"RLM {strategy} + {retrieval}"
        
        # Collect metrics
        # Some older result files might not have 'oolong_score', fallback to 0 or calc
        stats[label]['oolong'].append(r.get('oolong_score', 0.0))
        stats[label]['f1'].append(r.get('f1_score', 0.0))
        stats[label]['em'].append(1.0 if r.get('exact_match') else 0.0)
        stats[label]['time'].append(r.get('elapsed_time', 0.0))
        # Total calls = root + child
        total_calls = r.get('total_llm_calls', r.get('llm_calls', 0))
        stats[label]['calls'].append(total_calls)

    # Calculate averages
    labels = []
    avg_oolong = []
    avg_f1 = []
    avg_em = []
    avg_time = []
    avg_calls = []
    
    # Custom sort order
    def sort_key(label):
        if "Direct" in label: return (0, label)
        if "REPL Default" in label: return (1, label)
        if "No Recursion" in label: return (2, label)
        # Sort RLM strategies alphabetically
        return (3, label)

    for label in sorted(stats.keys(), key=sort_key):
        metrics = stats[label]
        labels.append(label)
        avg_oolong.append(np.mean(metrics['oolong']))
        avg_f1.append(np.mean(metrics['f1']))
        avg_em.append(np.mean(metrics['em']))
        avg_time.append(np.mean(metrics['time']))
        avg_calls.append(np.mean(metrics['calls']))

    # Plot 1: Scores (Only F1 Score)
    x = np.arange(len(labels))
    width = 0.6  # Wider bars since only one metric

    fig, ax = plt.subplots(figsize=(14, 7))
    # Only plot F1 score
    rects1 = ax.bar(x, avg_f1, width, label='F1 Score', color='tab:green')

    ax.set_ylabel('F1 Score')
    ax.set_title('OOLONGBench F1 Performance by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels for F1 score
    # Only label non-zero values
    f1_labels = [f'{v:.3f}' if v > 0.001 else '' for v in avg_f1]
    ax.bar_label(rects1, labels=f1_labels, padding=3)

    plt.tight_layout()
    plt.savefig(results_path / 'performance_comparison.png')
    print(f"Saved performance figure to {results_path / 'performance_comparison.png'}")

    # Plot 2: Efficiency (Time vs Calls)
    # Bar chart for Time
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Average Time (s)', color=color)
    ax1.bar(labels, avg_time, color=color, alpha=0.6, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Scatter points for Calls (no line)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average LLM Calls', color=color)
    # Remove linestyle to get rid of the line
    ax2.plot(labels, avg_calls, color=color, marker='o', linestyle='', markersize=8, label='LLM Calls')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Efficiency: Latency vs LLM Calls')
    fig.tight_layout()
    plt.savefig(results_path / 'efficiency_comparison.png')
    print(f"Saved efficiency figure to {results_path / 'efficiency_comparison.png'}")

if __name__ == "__main__":
    generate_figures()

