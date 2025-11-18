"""
Analysis script for OOLONGBench evaluation results.

This script loads results from oolongbench_evaluation.py and generates:
- Comparison tables
- Performance metrics
- Visualizations (if matplotlib is available)
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


class ResultsAnalyzer:
    """Analyzer for OOLONGBench evaluation results."""
    
    def __init__(self, results_dir: str = "oolongbench_results"):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.configs = {}
        
    def load_results(self):
        """Load all result files from the results directory."""
        print(f"Loading results from {self.results_dir}...")
        
        if not self.results_dir.exists():
            print(f"❌ Directory {self.results_dir} does not exist!")
            return
        
        # Load individual result files (not summary files)
        for file in self.results_dir.glob("*.json"):
            if file.name.startswith("summary"):
                continue
            
            with open(file) as f:
                results = json.load(f)
            
            # Extract config name
            # Format: {partition}_{retrieval}_parallel={bool}_{timestamp}.json
            parts = file.stem.split('_')
            if len(parts) >= 3:
                config_name = '_'.join(parts[:3])
            else:
                config_name = file.stem
            
            if config_name not in self.configs:
                self.configs[config_name] = []
            
            self.configs[config_name].extend(results)
        
        print(f"✓ Loaded {len(self.configs)} configurations")
        for config, results in self.configs.items():
            print(f"  - {config}: {len(results)} examples")
    
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregated metrics for each configuration.
        
        Returns:
            Dictionary mapping config names to their metrics
        """
        metrics = {}
        
        for config_name, results in self.configs.items():
            successful = [r for r in results if r.get('success', False)]
            
            if not successful:
                metrics[config_name] = {
                    'success_rate': 0.0,
                    'count': len(results),
                    'successful_count': 0
                }
                continue
            
            # Compute averages
            total_time = sum(r['elapsed_time'] for r in successful)
            total_calls = sum(r['llm_calls'] for r in successful)
            total_iterations = sum(r['iterations'] for r in successful)
            
            metrics[config_name] = {
                'success_rate': len(successful) / len(results),
                'count': len(results),
                'successful_count': len(successful),
                'avg_time': total_time / len(successful),
                'total_time': total_time,
                'avg_llm_calls': total_calls / len(successful),
                'total_llm_calls': total_calls,
                'avg_iterations': total_iterations / len(successful),
                'min_time': min(r['elapsed_time'] for r in successful),
                'max_time': max(r['elapsed_time'] for r in successful),
            }
        
        return metrics
    
    def print_comparison_table(self, metrics: Dict[str, Dict[str, float]]):
        """Print a comparison table of all configurations."""
        print("\n" + "="*100)
        print("CONFIGURATION COMPARISON")
        print("="*100)
        
        # Header
        print(f"{'Configuration':<40} {'Success':<10} {'Avg Time':<12} {'Avg Calls':<12} {'Avg Iters':<12}")
        print("-"*100)
        
        # Sort by avg time
        sorted_configs = sorted(
            metrics.items(),
            key=lambda x: x[1].get('avg_time', float('inf'))
        )
        
        for config_name, m in sorted_configs:
            success_str = f"{m['successful_count']}/{m['count']}"
            avg_time = f"{m.get('avg_time', 0):.2f}s"
            avg_calls = f"{m.get('avg_llm_calls', 0):.1f}"
            avg_iters = f"{m.get('avg_iterations', 0):.1f}"
            
            print(f"{config_name:<40} {success_str:<10} {avg_time:<12} {avg_calls:<12} {avg_iters:<12}")
        
        print("="*100 + "\n")
    
    def print_detailed_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Print detailed metrics for each configuration."""
        print("\n" + "="*100)
        print("DETAILED METRICS BY CONFIGURATION")
        print("="*100 + "\n")
        
        for config_name, m in sorted(metrics.items()):
            print(f"Configuration: {config_name}")
            print("-"*60)
            print(f"  Total examples: {m['count']}")
            print(f"  Successful: {m['successful_count']} ({m['success_rate']*100:.1f}%)")
            
            if m['successful_count'] > 0:
                print(f"  Average time: {m['avg_time']:.2f}s")
                print(f"  Total time: {m['total_time']:.2f}s")
                print(f"  Time range: {m['min_time']:.2f}s - {m['max_time']:.2f}s")
                print(f"  Average LLM calls: {m['avg_llm_calls']:.1f}")
                print(f"  Total LLM calls: {m['total_llm_calls']:.0f}")
                print(f"  Average iterations: {m['avg_iterations']:.1f}")
            print()
    
    def analyze_by_strategy(self, metrics: Dict[str, Dict[str, float]]):
        """Analyze results grouped by partition strategy."""
        print("\n" + "="*100)
        print("ANALYSIS BY PARTITION STRATEGY")
        print("="*100 + "\n")
        
        # Group by partition strategy (first part of config name)
        strategies = defaultdict(list)
        for config_name, m in metrics.items():
            strategy = config_name.split('_')[0]
            strategies[strategy].append((config_name, m))
        
        for strategy, configs in strategies.items():
            print(f"Partition Strategy: {strategy.upper()}")
            print("-"*60)
            
            for config_name, m in configs:
                if m['successful_count'] > 0:
                    print(f"  {config_name}:")
                    print(f"    Avg time: {m['avg_time']:.2f}s")
                    print(f"    Avg calls: {m['avg_llm_calls']:.1f}")
            print()
    
    def find_best_config(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Find the best configuration based on combined metrics.
        
        Returns:
            Name of the best configuration
        """
        # Filter successful configs
        valid_configs = {
            name: m for name, m in metrics.items()
            if m['successful_count'] > 0
        }
        
        if not valid_configs:
            return None
        
        # Best by time (fastest)
        best_time = min(valid_configs.items(), key=lambda x: x[1]['avg_time'])
        
        # Best by efficiency (fewest LLM calls)
        best_efficiency = min(valid_configs.items(), key=lambda x: x[1]['avg_llm_calls'])
        
        print("\n" + "="*100)
        print("BEST CONFIGURATIONS")
        print("="*100)
        print(f"\nFastest (lowest avg time):")
        print(f"  {best_time[0]}: {best_time[1]['avg_time']:.2f}s")
        
        print(f"\nMost efficient (fewest LLM calls):")
        print(f"  {best_efficiency[0]}: {best_efficiency[1]['avg_llm_calls']:.1f} calls")
        print()
        
        return best_time[0]
    
    def export_to_csv(self, metrics: Dict[str, Dict[str, float]], output_file: str = "results.csv"):
        """Export metrics to CSV file."""
        import csv
        
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Configuration',
                'Total Examples',
                'Successful',
                'Success Rate',
                'Avg Time (s)',
                'Total Time (s)',
                'Avg LLM Calls',
                'Total LLM Calls',
                'Avg Iterations'
            ])
            
            # Data
            for config_name, m in sorted(metrics.items()):
                writer.writerow([
                    config_name,
                    m['count'],
                    m['successful_count'],
                    f"{m['success_rate']*100:.1f}%",
                    f"{m.get('avg_time', 0):.2f}",
                    f"{m.get('total_time', 0):.2f}",
                    f"{m.get('avg_llm_calls', 0):.1f}",
                    f"{m.get('total_llm_calls', 0):.0f}",
                    f"{m.get('avg_iterations', 0):.1f}"
                ])
        
        print(f"✓ Exported results to {output_path}")
    
    def plot_results(self, metrics: Dict[str, Dict[str, float]]):
        """Generate visualization plots (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠ matplotlib not installed. Skipping plots.")
            print("  Install with: pip install matplotlib")
            return
        
        # Filter successful configs
        valid_configs = {
            name: m for name, m in metrics.items()
            if m['successful_count'] > 0
        }
        
        if not valid_configs:
            print("No successful configurations to plot")
            return
        
        configs = list(valid_configs.keys())
        avg_times = [valid_configs[c]['avg_time'] for c in configs]
        avg_calls = [valid_configs[c]['avg_llm_calls'] for c in configs]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Average Time
        ax1.bar(range(len(configs)), avg_times)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Average Time (s)')
        ax1.set_title('Average Processing Time by Configuration')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Average LLM Calls
        ax2.bar(range(len(configs)), avg_calls, color='orange')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Average LLM Calls')
        ax2.set_title('Average LLM Calls by Configuration')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")
        
        # Optionally show plot
        # plt.show()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*100)
        print("OOLONGBENCH RESULTS ANALYSIS")
        print("="*100 + "\n")
        
        # Load results
        self.load_results()
        
        if not self.configs:
            print("❌ No results found!")
            return
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Print analyses
        self.print_comparison_table(metrics)
        self.print_detailed_metrics(metrics)
        self.analyze_by_strategy(metrics)
        self.find_best_config(metrics)
        
        # Export results
        self.export_to_csv(metrics)
        
        # Generate plots
        self.plot_results(metrics)
        
        print("\n✓ Analysis complete!")


def main():
    """Main function."""
    analyzer = ResultsAnalyzer(results_dir="oolongbench_results")
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()

