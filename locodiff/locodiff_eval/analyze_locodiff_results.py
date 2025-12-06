"""
Analysis script for LoCoDiff evaluation results.

This script loads results from locodiff_evaluation.py and generates:
- Accuracy metrics based on similarity thresholds (primary)
- Exact match metrics (secondary/reference)
- Performance by context length (key LoCoDiff insight)
- Performance by language and repository
- RLM vs Baseline comparison
- Visualizations
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


class LoCoDiffResultsAnalyzer:
    """Analyzer for LoCoDiff evaluation results."""

    def __init__(self, results_dir: str = "locodiff_results", similarity_threshold: float = 0.90):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing result JSON files
            similarity_threshold: Similarity needed to count as accurate
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
        self.similarity_threshold = similarity_threshold

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
            # Format: {method}_{partition}_{retrieval}_parallel={bool}_{timestamp}.json
            parts = file.stem.split('_')

            # Handle baseline separately
            if parts[0] == "baseline":
                config_name = "baseline"
            elif len(parts) >= 4:
                # rlm_token_unfiltered_parallel=False
                config_name = '_'.join(parts[:4])
            else:
                config_name = file.stem

            # Normalize name for the primary RLM config
            if config_name.startswith("rlm_none_unfiltered"):
                config_name = "RLM"

            if config_name not in self.configs:
                self.configs[config_name] = []

            self.configs[config_name].extend(results)

        print(f"✓ Loaded {len(self.configs)} configurations")
        for config, results in self.configs.items():
            successful = sum(1 for r in results if r.get('success'))
            print(f"  - {config}: {len(results)} examples ({successful} successful)")

    def compute_metrics(self) -> Dict[str, Dict[str, Any]]:
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
                    'successful_count': 0,
                    'accuracy_threshold': self.similarity_threshold,
                    'accurate_count': 0,
                    'accuracy_rate': 0.0,
                    'exact_match_count': 0,
                    'exact_match_accuracy': 0.0,
                    'avg_similarity': 0.0,
                    'avg_time': 0.0,
                    'total_time': 0.0,
                    'avg_llm_calls': 0.0,
                    'total_llm_calls': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0
                }
                continue

            # Compute accuracy (primary metric) and exact match (reference)
            accurate = sum(1 for r in successful if r.get('similarity', 0) >= self.similarity_threshold)
            accuracy_rate = accurate / len(successful)
            exact_matches = sum(1 for r in successful if r.get('exact_match', False))
            exact_match_accuracy = exact_matches / len(successful)

            # Compute averages
            total_time = sum(r['elapsed_time'] for r in successful)
            total_calls = sum(r['llm_calls'] for r in successful)
            avg_similarity = sum(r.get('similarity', 0) for r in successful) / len(successful)

            metrics[config_name] = {
                'success_rate': len(successful) / len(results),
                'count': len(results),
                'successful_count': len(successful),
                'accuracy_threshold': self.similarity_threshold,
                'accurate_count': accurate,
                'accuracy_rate': accuracy_rate,
                'exact_match_count': exact_matches,
                'exact_match_accuracy': exact_match_accuracy,
                'avg_similarity': avg_similarity,
                'avg_time': total_time / len(successful),
                'total_time': total_time,
                'avg_llm_calls': total_calls / len(successful),
                'total_llm_calls': total_calls,
                'min_time': min(r['elapsed_time'] for r in successful),
                'max_time': max(r['elapsed_time'] for r in successful),
            }

        return metrics

    def analyze_by_context_length(
        self,
        bucket_size: int = 10000
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results by context length (token count).
        This is a key LoCoDiff insight: accuracy degrades with context length.

        Args:
            bucket_size: Size of token buckets (default: 10k)

        Returns:
            Dictionary mapping config -> bucket -> metrics
        """
        print("\n" + "="*100)
        print("ANALYSIS BY CONTEXT LENGTH")
        print("="*100 + "\n")

        analysis = {}

        for config_name, results in self.configs.items():
            # Bucket results by token count
            buckets = defaultdict(list)

            for r in results:
                if not r.get('success'):
                    continue

                tokens = r.get('prompt_tokens', 0)
                bucket = (tokens // bucket_size) * bucket_size
                buckets[bucket].append(r)

            # Compute metrics per bucket
            bucket_metrics = {}
            for bucket, bucket_results in sorted(buckets.items()):
                accurate = sum(1 for r in bucket_results if r.get('similarity', 0) >= self.similarity_threshold)
                accuracy = accurate / len(bucket_results)

                bucket_metrics[f"{bucket//1000}k-{(bucket+bucket_size)//1000}k"] = {
                    'count': len(bucket_results),
                    'accurate_count': accurate,
                    'accuracy_rate': accuracy,
                    'avg_similarity': sum(r.get('similarity', 0) for r in bucket_results) / len(bucket_results)
                }

            analysis[config_name] = bucket_metrics

        # Print analysis
        for config_name, bucket_metrics in analysis.items():
            print(f"\n{config_name}:")
            print(f"{'Token Range':<15} {'Count':<8} {'Accurate':<15} {'Accuracy':<12} {'Avg Similarity':<15}")
            print("-"*80)

            for bucket_name, metrics in bucket_metrics.items():
                accurate_str = f"{metrics['accurate_count']}/{metrics['count']}"
                acc_str = f"{metrics['accuracy_rate']:.1%}"
                sim_str = f"{metrics['avg_similarity']:.1%}"

                print(f"{bucket_name:<15} {metrics['count']:<8} {accurate_str:<15} {acc_str:<12} {sim_str:<15}")

        return analysis

    def analyze_by_language(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results by programming language.

        Returns:
            Dictionary mapping config -> language -> metrics
        """
        print("\n" + "="*100)
        print("ANALYSIS BY PROGRAMMING LANGUAGE")
        print("="*100 + "\n")

        analysis = {}

        for config_name, results in self.configs.items():
            # Group by language
            languages = defaultdict(list)

            for r in results:
                if not r.get('success'):
                    continue

                lang = r.get('language', 'Unknown')
                languages[lang].append(r)

            # Compute metrics per language
            lang_metrics = {}
            for lang, lang_results in languages.items():
                accurate = sum(1 for r in lang_results if r.get('similarity', 0) >= self.similarity_threshold)
                accuracy = accurate / len(lang_results)

                lang_metrics[lang] = {
                    'count': len(lang_results),
                    'accurate_count': accurate,
                    'accuracy_rate': accuracy,
                    'avg_time': sum(r.get('elapsed_time', 0) for r in lang_results) / len(lang_results)
                }

            analysis[config_name] = lang_metrics

        # Print analysis
        for config_name, lang_metrics in analysis.items():
            print(f"\n{config_name}:")
            print(f"{'Language':<15} {'Count':<8} {'Accurate':<15} {'Accuracy':<12} {'Avg Time':<12}")
            print("-"*70)

            for lang, metrics in sorted(lang_metrics.items(), key=lambda x: x[1]['accuracy_rate'], reverse=True):
                accurate_str = f"{metrics['accurate_count']}/{metrics['count']}"
                acc_str = f"{metrics['accuracy_rate']:.1%}"
                time_str = f"{metrics['avg_time']:.2f}s"

                print(f"{lang:<15} {metrics['count']:<8} {accurate_str:<15} {acc_str:<12} {time_str:<12}")

        return analysis

    def analyze_by_repo(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results by repository.

        Returns:
            Dictionary mapping config -> repo -> metrics
        """
        print("\n" + "="*100)
        print("ANALYSIS BY REPOSITORY")
        print("="*100 + "\n")

        analysis = {}

        for config_name, results in self.configs.items():
            # Group by repo
            repos = defaultdict(list)

            for r in results:
                if not r.get('success'):
                    continue

                repo = r.get('repo', 'unknown')
                repos[repo].append(r)

            # Compute metrics per repo
            repo_metrics = {}
            for repo, repo_results in repos.items():
                accurate = sum(1 for r in repo_results if r.get('similarity', 0) >= self.similarity_threshold)
                accuracy = accurate / len(repo_results)

                repo_metrics[repo] = {
                    'count': len(repo_results),
                    'accurate_count': accurate,
                    'accuracy_rate': accuracy
                }

            analysis[config_name] = repo_metrics

        # Print analysis
        for config_name, repo_metrics in analysis.items():
            print(f"\n{config_name}:")
            print(f"{'Repository':<20} {'Count':<8} {'Accurate':<15} {'Accuracy':<12}")
            print("-"*60)

            for repo, metrics in sorted(repo_metrics.items()):
                accurate_str = f"{metrics['accurate_count']}/{metrics['count']}"
                acc_str = f"{metrics['accuracy_rate']:.1%}"

                print(f"{repo:<20} {metrics['count']:<8} {accurate_str:<15} {acc_str:<12}")

        return analysis

    def compare_rlm_vs_baseline(self, metrics: Dict[str, Dict[str, Any]]):
        """
        Compare RLM configurations against baseline.

        Args:
            metrics: Computed metrics for all configurations
        """
        print("\n" + "="*100)
        print("RLM vs BASELINE COMPARISON")
        print("="*100 + "\n")

        baseline_metrics = metrics.get('baseline')

        if not baseline_metrics:
            print("❌ No baseline results found for comparison")
            return

        print(f"BASELINE:")
        print(f"  Accuracy (sim ≥ {self.similarity_threshold:.0%}): {baseline_metrics['accuracy_rate']:.1%} ({baseline_metrics['accurate_count']}/{baseline_metrics['count']})")
        print(f"  Exact Match Accuracy: {baseline_metrics['exact_match_accuracy']:.1%} ({baseline_metrics['exact_match_count']}/{baseline_metrics['count']})")
        print(f"  Avg Time: {baseline_metrics['avg_time']:.2f}s")
        print(f"  Avg LLM Calls: {baseline_metrics['avg_llm_calls']:.1f}")
        print()

        print(f"{'Configuration':<50} {'Accuracy':<15} {'vs Baseline':<15} {'Avg Time':<12} {'Avg Calls':<12}")
        print("-"*110)

        # Sort by accuracy
        rlm_configs = {k: v for k, v in metrics.items() if k != 'baseline'}
        sorted_configs = sorted(
            rlm_configs.items(),
            key=lambda x: x[1]['accuracy_rate'],
            reverse=True
        )

        for config_name, m in sorted_configs:
            acc_str = f"{m['accuracy_rate']:.1%}"

            # Compare to baseline
            accuracy_diff = m['accuracy_rate'] - baseline_metrics['accuracy_rate']
            diff_str = f"{accuracy_diff:+.1%}"

            time_str = f"{m['avg_time']:.2f}s"
            calls_str = f"{m['avg_llm_calls']:.1f}"

            print(f"{config_name:<50} {acc_str:<15} {diff_str:<15} {time_str:<12} {calls_str:<12}")

    def print_comparison_table(self, metrics: Dict[str, Dict[str, float]]):
        """Print a comparison table of all configurations."""
        print("\n" + "="*110)
        print("CONFIGURATION COMPARISON")
        print("="*110)

        # Header
        print(f"{'Configuration':<50} {'Success':<10} {'Accurate':<15} {'Accuracy':<12} {'Avg Time':<12} {'Avg Calls':<12}")
        print("-"*110)

        # Sort by accuracy
        sorted_configs = sorted(
            metrics.items(),
            key=lambda x: x[1].get('accuracy_rate', 0),
            reverse=True
        )

        for config_name, m in sorted_configs:
            success_str = f"{m['successful_count']}/{m['count']}"
            accurate_str = f"{m.get('accurate_count', 0)}/{m['successful_count']}" if m['successful_count'] > 0 else "0/0"
            acc_str = f"({m.get('accuracy_rate', 0):.1%})"
            avg_time = f"{m.get('avg_time', 0):.2f}s"
            avg_calls = f"{m.get('avg_llm_calls', 0):.1f}"

            print(f"{config_name:<50} {success_str:<10} {accurate_str:<8} {acc_str:<7} {avg_time:<12} {avg_calls:<12}")

        print("="*110 + "\n")

    def find_best_config(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Find the best configuration based on accuracy (similarity threshold).

        Args:
            metrics: Computed metrics

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

        # Best by accuracy
        best_accuracy = max(valid_configs.items(), key=lambda x: x[1]['accuracy_rate'])

        # Best by time (among configs with > 50% accuracy)
        high_accuracy_configs = {
            name: m for name, m in valid_configs.items()
            if m['accuracy_rate'] > 0.5
        }

        if high_accuracy_configs:
            best_time = min(high_accuracy_configs.items(), key=lambda x: x[1]['avg_time'])
        else:
            best_time = None

        print("\n" + "="*100)
        print("BEST CONFIGURATIONS")
        print("="*100)
        print(f"\nHighest Accuracy (sim ≥ {self.similarity_threshold:.0%}):")
        print(f"  {best_accuracy[0]}: {best_accuracy[1]['accuracy_rate']:.1%} accuracy")

        if best_time:
            print(f"\nFastest (among >50% accuracy):")
            print(f"  {best_time[0]}: {best_time[1]['avg_time']:.2f}s avg time")
        print()

        return best_accuracy[0]

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
                'Accurate Count',
                'Accuracy Rate',
                'Exact Match Count',
                'Exact Match Accuracy',
                'Avg Similarity',
                'Avg Time (s)',
                'Total Time (s)',
                'Avg LLM Calls',
                'Total LLM Calls'
            ])

            # Data
            for config_name, m in sorted(metrics.items()):
                writer.writerow([
                    config_name,
                    m['count'],
                    m['successful_count'],
                    f"{m['success_rate']*100:.1f}%",
                    m.get('accurate_count', 0),
                    f"{m.get('accuracy_rate', 0)*100:.1f}%",
                    m.get('exact_match_count', 0),
                    f"{m.get('exact_match_accuracy', 0)*100:.1f}%",
                    f"{m.get('avg_similarity', 0)*100:.1f}%",
                    f"{m.get('avg_time', 0):.2f}",
                    f"{m.get('total_time', 0):.2f}",
                    f"{m.get('avg_llm_calls', 0):.1f}",
                    f"{m.get('total_llm_calls', 0):.0f}"
                ])

        print(f"✓ Exported results to {output_path}")

    def plot_accuracy_vs_context(self, context_analysis: Dict[str, Dict[str, Any]]):
        """
        Plot accuracy vs context length (key LoCoDiff visualization).

        Args:
            context_analysis: Results from analyze_by_context_length()
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠ matplotlib not installed. Skipping plots.")
            print("  Install with: pip install matplotlib")
            return

        plt.figure(figsize=(12, 6))

        for config_name, bucket_metrics in context_analysis.items():
            # Extract buckets and accuracies
            buckets = []
            accuracies = []

            for bucket_name, metrics in sorted(bucket_metrics.items()):
                # Extract start token count from bucket name (e.g., "0k-10k" -> 0)
                start_k = int(bucket_name.split('k')[0])
                buckets.append(start_k)
                accuracies.append(metrics['accuracy_rate'] * 100)

            # Plot line
            label = config_name.replace('rlm_', '').replace('_parallel=False', '')
            plt.plot(buckets, accuracies, marker='o', label=label, linewidth=2)

        plt.xlabel('Context Length (k tokens)', fontsize=12)
        plt.ylabel(f'Accuracy (sim ≥ {self.similarity_threshold:.0%}) (%)', fontsize=12)
        plt.title('LoCoDiff: Accuracy vs Context Length', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / 'accuracy_vs_context.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")

        plt.close()

    def plot_comparison(self, metrics: Dict[str, Dict[str, float]]):
        """Generate comparison bar charts."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠ matplotlib not installed. Skipping plots.")
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
        accuracies = [valid_configs[c]['accuracy_rate'] * 100 for c in configs]
        avg_times = [valid_configs[c]['avg_time'] for c in configs]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Accuracy (similarity-based)
        colors = ['green' if c == 'baseline' else 'blue' for c in configs]
        ax1.bar(range(len(configs)), accuracies, color=colors, alpha=0.7)
        ax1.set_xlabel('Configuration', fontsize=11)
        ax1.set_ylabel(f'Accuracy (sim ≥ {self.similarity_threshold:.0%}) (%)', fontsize=11)
        ax1.set_title('Accuracy by Configuration', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels([c.replace('rlm_', '').replace('_parallel=False', '') for c in configs],
                            rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Average Time
        ax2.bar(range(len(configs)), avg_times, color='orange', alpha=0.7)
        ax2.set_xlabel('Configuration', fontsize=11)
        ax2.set_ylabel('Average Time (s)', fontsize=11)
        ax2.set_title('Average Processing Time by Configuration', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels([c.replace('rlm_', '').replace('_parallel=False', '') for c in configs],
                            rotation=45, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")

        plt.close()

    def plot_similarity_distribution(self):
        """Plot similarity distribution for each configuration."""
        if not self.configs:
            print("No configs loaded; skipping similarity distribution plot.")
            return

        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 1, 31)

        for config_name, results in self.configs.items():
            sims = [r.get('similarity', 0) for r in results if r.get('success')]
            if not sims:
                continue
            plt.hist(sims, bins=bins, alpha=0.4, label=config_name, density=True)

        plt.xlabel('Similarity', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Similarity Distribution by Configuration', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        plot_path = self.plots_dir / 'similarity_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {plot_path}")
        plt.close()

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*100)
        print("LOCODIFF RESULTS ANALYSIS")
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
        self.compare_rlm_vs_baseline(metrics)

        # Analyze by different dimensions
        context_analysis = self.analyze_by_context_length()
        self.analyze_by_language()
        self.analyze_by_repo()

        # Find best config
        self.find_best_config(metrics)

        # Export results
        self.export_to_csv(metrics)

        # Generate plots
        self.plot_accuracy_vs_context(context_analysis)
        self.plot_comparison(metrics)
        self.plot_similarity_distribution()

        print("\n✓ Analysis complete!")


def main():
    """Main function."""
    analyzer = LoCoDiffResultsAnalyzer(results_dir="locodiff_results")
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
