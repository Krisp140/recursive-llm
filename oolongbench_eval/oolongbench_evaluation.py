"""
OOLONGBench Evaluation Script for RLM with Partition Strategies

This script evaluates RLM on the OOLONGBench dataset with different partition strategies.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from rlm import RLM


class OOLONGBenchEvaluator:
    """Evaluator for running RLM on OOLONGBench with different partition strategies."""
    
    def __init__(
        self,
        model: str = "gpt-5-mini",
        recursive_model: str = "gpt-5-mini",
        output_dir: str = "oolongbench_results"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Root LLM model name
            recursive_model: Recursive LLM model name
            output_dir: Directory to save results
        """
        self.model = model
        self.recursive_model = recursive_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_oolongbench(self, config: str = "dnd"):
        """
        Load OOLONGBench dataset from Hugging Face.
        
        Args:
            config: Dataset config to load. Options:
                   - 'dnd': Full D&D dataset (recommended for evaluation)
                   - 'toy_dnd': Smaller toy dataset for testing
        
        Returns:
            Dataset object
        """
        print(f"Loading OOLONGBench dataset from Hugging Face (config: {config})...")
        
        try:
            # OOLONGBench requires a config name: 'dnd' or 'toy_dnd'
            dataset = load_dataset("oolongbench/oolong-real", config)
            print(f"✓ Loaded OOLONGBench successfully (config: {config})")
            print(f"  Available splits: {list(dataset.keys())}")
            return dataset
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("\nAvailable configs: 'dnd', 'toy_dnd'")
            print("Tip: Use 'toy_dnd' for faster testing")
            raise
    
    def calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate F1 score between prediction and ground truth.
        Simple token overlap F1.
        """
        def normalize_text(s):
            import re
            import string
            if not s:
                return ""
            s = s.lower()
            # Remove punctuation
            s = "".join(c for c in s if c not in string.punctuation)
            # Remove extra whitespace
            s = " ".join(s.split())
            return s

        pred_tokens = normalize_text(prediction).split()
        truth_tokens = normalize_text(ground_truth).split()
        
        if not pred_tokens or not truth_tokens:
            return 0.0
            
        common = set(pred_tokens) & set(truth_tokens)
        num_same = len(common)
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall)

    async def evaluate_single_example(
        self,
        example: Dict[str, Any],
        partition_strategy: str,
        retrieval_method: str,
        parallel_subqueries: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single example from OOLONGBench.
        
        Args:
            example: Dataset example with 'context' and 'question' fields
            partition_strategy: Partition strategy to use
            retrieval_method: Retrieval method to use
            parallel_subqueries: Whether to use parallel subqueries
        
        Returns:
            Results dictionary with metrics
        """
        # Initialize RLM with partition strategy
        rlm = RLM(
            model=self.model,
            recursive_model=self.recursive_model,
            partition_strategy=partition_strategy,
            retrieval_method=retrieval_method,
            parallel_subqueries=parallel_subqueries,
            max_iterations=30,
            max_depth=5,
            temperature=1.0  # GPT-5 models require temperature=1
        )
        
        # Extract context and question from example
        # OOLONGBench uses: context_window_text, question, answer
        context = example.get('context_window_text', example.get('context', example.get('input', '')))
        question = example.get('question', example.get('query', ''))
        ground_truth = example.get('answer', example.get('output', ''))
        
        # Debug: Check if we have valid data
        if not context or not question:
            return {
                'success': False,
                'error': f"Missing data - context length: {len(context)}, question length: {len(question)}",
                'question': question,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries,
                'f1_score': 0.0,
                'exact_match': False
            }
        
        # Measure time and tokens
        start_time = time.time()
        
        try:
            # Run RLM completion
            answer = await rlm.acompletion(query=question, context=context)
            
            elapsed_time = time.time() - start_time
            
            # Calculate scores
            f1_score = self.calculate_f1(answer, ground_truth)
            exact_match = answer.strip().lower() == ground_truth.strip().lower() if ground_truth else False

            # Get stats
            stats = rlm.stats
            
            return {
                'success': True,
                'answer': answer,
                'ground_truth': ground_truth,
                'f1_score': f1_score,
                'exact_match': exact_match,
                'context_length': len(context),
                'question': question,
                'llm_calls': stats['llm_calls'],
                'iterations': stats['iterations'],
                'depth': stats['depth'],
                'elapsed_time': elapsed_time,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries,
                'f1_score': 0.0,
                'exact_match': False
            }
    
    async def evaluate_dataset(
        self,
        dataset,
        partition_strategies: List[str],
        retrieval_methods: List[str],
        parallel_options: List[bool],
        max_examples: int = None
    ):
        """
        Evaluate RLM on OOLONGBench with different configurations.
        
        Args:
            dataset: OOLONGBench dataset
            partition_strategies: List of partition strategies to test
            retrieval_methods: List of retrieval methods to test
            parallel_options: List of parallel subquery options [False, True]
            max_examples: Maximum number of examples to evaluate (None = all)
        """
        # Get examples from dataset
        # Handle different dataset structures
        if hasattr(dataset, 'keys'):
            # Dataset has splits like 'train', 'test', 'validation'
            examples = list(dataset['test']) if 'test' in dataset else list(dataset['train'])
        else:
            examples = list(dataset)
        
        if max_examples:
            examples = examples[:max_examples]
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(examples)} examples")
        print(f"{'='*60}\n")
        
        all_results = []
        
        # Iterate over all configuration combinations
        for partition_strategy in partition_strategies:
            for retrieval_method in retrieval_methods:
                for parallel in parallel_options:
                    
                    config_name = f"{partition_strategy}_{retrieval_method}_parallel={parallel}"
                    print(f"\n{'='*60}")
                    print(f"Configuration: {config_name}")
                    print(f"{'='*60}")
                    
                    config_results = []
                    
                    for i, example in enumerate(examples, 1):
                        print(f"\nExample {i}/{len(examples)}")
                        
                        result = await self.evaluate_single_example(
                            example,
                            partition_strategy,
                            retrieval_method,
                            parallel
                        )
                        
                        config_results.append(result)
                        
                        if result['success']:
                            print(f"  ✓ Success - Time: {result['elapsed_time']:.2f}s, "
                                  f"LLM calls: {result['llm_calls']}")
                        else:
                            print(f"  ✗ Failed - {result['error']}")
                    
                    # Save configuration results
                    self.save_results(config_name, config_results)
                    all_results.extend(config_results)
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def save_results(self, config_name: str, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        output_file = self.output_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved results to {output_file}")
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate and save summary statistics."""
        # Group results by configuration
        configs = {}
        for result in all_results:
            if not result['success']:
                continue
            
            config_key = (
                result['partition_strategy'],
                result['retrieval_method'],
                result['parallel_subqueries']
            )
            
            if config_key not in configs:
                configs[config_key] = {
                    'results': [],
                    'total_time': 0,
                    'total_llm_calls': 0,
                    'total_iterations': 0
                }
            
            configs[config_key]['results'].append(result)
            configs[config_key]['total_time'] += result['elapsed_time']
            configs[config_key]['total_llm_calls'] += result['llm_calls']
            configs[config_key]['total_iterations'] += result['iterations']
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'recursive_model': self.recursive_model,
            'configurations': {}
        }
        
        for config_key, data in configs.items():
            partition_strategy, retrieval_method, parallel = config_key
            config_name = f"{partition_strategy}_{retrieval_method}_parallel={parallel}"
            
            num_examples = len(data['results'])
            
            summary['configurations'][config_name] = {
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel,
                'num_examples': num_examples,
                'avg_time': data['total_time'] / num_examples,
                'avg_llm_calls': data['total_llm_calls'] / num_examples,
                'avg_iterations': data['total_iterations'] / num_examples,
                'total_time': data['total_time']
            }
        
        # Save summary
        summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SUMMARY REPORT")
        print(f"{'='*60}\n")
        
        for config_name, metrics in summary['configurations'].items():
            print(f"{config_name}:")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  Avg LLM Calls: {metrics['avg_llm_calls']:.1f}")
            print(f"  Avg Iterations: {metrics['avg_iterations']:.1f}")
            print()
        
        print(f"✓ Saved summary to {summary_file}")


async def main():
    """Main evaluation function."""
    
    # Initialize evaluator
    evaluator = OOLONGBenchEvaluator(
        model="gpt-5-mini",
        recursive_model="gpt-5-mini",
        output_dir="oolongbench_results"
    )
    
    # Load OOLONGBench dataset
    # Options: 'dnd' (full dataset) or 'toy_dnd' (smaller, for quick testing)
    dataset = evaluator.load_oolongbench(config="toy_dnd")  # Use 'toy_dnd' for faster testing
    
    # Print dataset structure
    print("\nDataset structure:")
    print(dataset)
    
    # Define configurations to test
    partition_strategies = [
        "token",        # Baseline: fixed-size chunks
        "structural",   # Now implemented
        "semantic",     # Now implemented (requires embedding model)
        # "learned"       # Not yet implemented
    ]
    
    retrieval_methods = [
        "unfiltered",  # No retrieval (baseline)
        "regex",       # Now implemented
        "embedding"    # Now implemented
    ]
    
    parallel_options = [
        False,  # Sequential processing
        # True    # Parallel not fully implemented in core yet
    ]
    
    # Run evaluation
    # Start with a small number for testing
    results = await evaluator.evaluate_dataset(
        dataset,
        partition_strategies,
        retrieval_methods,
        parallel_options,
        max_examples=5  # Change to 10, 20, or None (all examples)
    )
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved to {evaluator.output_dir}")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not found!")
        print("\nPlease set your API key:")
        print("  1. Create a .env file in the project root with:")
        print("     GEMINI_API_KEY=your-key-here")
        print("  2. Or export it in your shell:")
        print("     export GEMINI_API_KEY='your-key-here'")
        exit(1)
    
    # Install required packages if needed
    print("Make sure you have installed:")
    print("  pip install datasets huggingface-hub")
    print()
    
    asyncio.run(main())

