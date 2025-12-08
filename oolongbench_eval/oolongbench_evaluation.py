"""
OOLONGBench Evaluation Script for RLM with Partition Strategies

This script evaluates RLM on the OOLONGBench dataset with different partition strategies.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import litellm

# Try importing openai for semantic check
try:
    import openai
except ImportError:
    openai = None

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
        
        # Logic to handle 'trec_coarse' request
        if config == "trec_coarse":
             print("Loading OOLONG-synth and filtering for TREC-QC-coarse only...")
             try:
                 # TREC-QC-coarse is in the validation split of oolong-synth
                 dataset = load_dataset("oolongbench/oolong-synth", split="validation")
                 
                 # Optional: Inspect schema for debugging
                 if len(dataset) > 0:
                     print(f"  Sample keys: {list(dataset[0].keys())}")
                 
                 # Filter to only TREC-QC-coarse examples
                 def is_trec_coarse(example):
                     source = example.get('dataset', example.get('source_dataset', example.get('task', '')))
                     source_lower = str(source).lower().replace('-', '_').replace(' ', '_')
                     # Match variations: "trec_coarse", "trec_qc_coarse", "TREC-QC-coarse"
                     return 'trec' in source_lower and ('coarse' in source_lower or 'qc' in source_lower)
                 
                 # Optional: Filter by context length (roughly 132k tokens ~ 528k chars)
                 # Alex Zhang's evaluation used ~132k token contexts
                 def is_target_length(example):
                     # Target ~132k tokens. Using range 125k-140k tokens (approx 4 chars/token)
                     min_chars = 125000 * 4.68
                     max_chars = 140000 * 4.68
                     context = example.get('context_window_text', '')
                     return min_chars <= len(context) <= max_chars

                 filtered_dataset = dataset.filter(is_trec_coarse)
                 print(f"[OK] Found {len(filtered_dataset)} TREC-QC-coarse examples.")
                 
                 filtered_dataset = filtered_dataset.filter(is_target_length)
                 print(f"[OK] Filtered to {len(filtered_dataset)} examples with ~132k tokens.")
                 
                 return filtered_dataset
                 
             except Exception as e:
                 print(f"Failed to load/filter oolong-synth: {e}")
                 print("Falling back to 'oolongbench/oolong-real' (dnd) as proxy.")
                 config = "dnd" # Fallback

        try:
            # OOLONGBench requires a config name: 'dnd' or 'toy_dnd'
            # If config is invalid, it might fail.
            dataset = load_dataset("oolongbench/oolong-real", config)
            print(f"[OK] Loaded OOLONGBench successfully (config: {config})")
            print(f"  Available splits: {list(dataset.keys())}")
            return dataset
        except Exception as e:
            print(f"[ERROR] Error loading dataset: {e}")
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

    def calculate_numeric_error(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate percentage error if both prediction and ground truth are numeric.
        Returns -1.0 if non-numeric.
        """
        try:
            # Remove commas and handle basic cleaning
            import re
            def clean_num(s):
                if not s: return 0.0
                # Extract first number found
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", s)
                if not matches: return 0.0
                return float(matches[0])

            pred_val = clean_num(prediction)
            truth_val = clean_num(ground_truth)
            
            if truth_val == 0:
                return 0.0 if pred_val == 0 else float('inf')
                
            error = abs(pred_val - truth_val) / truth_val
            return error
        except:
            return -1.0

    def calculate_oolong_score(self, prediction: str, ground_truth: str, answer_type: Optional[str] = None) -> float:
        """
        OOLONG's scoring: score = 0.75^|y - ŷ| for numeric answers.
        Returns 1.0/0.0 for exact match if non-numeric or specific categorical types.
        """
        # Clean answer type string (e.g., 'ANSWER_TYPE.NUMERIC' -> 'numeric')
        if answer_type:
            answer_type = answer_type.lower().split('.')[-1]
            
        # Types that require strict exact match
        exact_match_types = {'label', 'comparison', 'user', 'date', 'month_year'}
        
        if answer_type in exact_match_types:
            return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0

        import re
        
        def extract_number(s):
            if not s:
                return None
            # Look for numbers, including decimals and negatives
            matches = re.findall(r"[-+]?\d*\.?\d+", str(s))
            return float(matches[0]) if matches else None
        
        pred = extract_number(prediction)
        truth = extract_number(ground_truth)
        
        if pred is None or truth is None:
            # Fall back to exact match for non-numeric
            return 1.0 if str(prediction).strip().lower() == str(ground_truth).strip().lower() else 0.0
        
        # Calculate OOLONG numeric score
        return 0.75 ** abs(truth - pred)

    async def evaluate_single_example(
        self,
        example: Dict[str, Any],
        partition_strategy: Optional[str],
        retrieval_method: Optional[str],
        parallel_subqueries: bool = False,
        mode: str = "rlm" # "rlm", "direct", "rlm-no-recursion"
    ) -> Dict[str, Any]:
        """
        Evaluate a single example from OOLONGBench.
        
        Args:
            example: Dataset example with 'context' and 'question' fields
            partition_strategy: Partition strategy to use
            retrieval_method: Retrieval method to use
            parallel_subqueries: Whether to use parallel subqueries
            mode: Evaluation mode: "rlm" (recursive), "direct" (no RLM), "rlm-no-recursion" (RLM without subcalls)
        
        Returns:
            Results dictionary with metrics
        """
        
        # Extract context and question from example
        # OOLONGBench uses: context_window_text, question, answer
        context = example.get('context_window_text', example.get('context', example.get('input', '')))
        question = example.get('question', example.get('query', ''))
        ground_truth = example.get('answer', example.get('output', ''))
        answer_type = example.get('answer_type')
        
        if not context or not question:
             return {
                'success': False,
                'error': f"Missing data",
                'question': question,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries,
                'mode': mode,
                'f1_score': 0.0,
                'oolong_score': 0.0,
                'exact_match': False
            }

        start_time = time.time()
        
        print(f"  -> Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"  -> Context: {len(context):,} chars")

        answer = ""
        llm_calls = 0
        child_llm_calls = 0
        iterations = 0
        depth = 0
        
        try:
            if mode == "direct":
                # Direct call to model using litellm
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the user question based on the context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
                
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=1.0
                )
                answer = response.choices[0].message.content
                llm_calls = 1
                
            elif mode == "rlm-no-recursion":
                # RLM with recursion disabled (ablation)
                # Setting max_depth=1 allows root to run but prevents any recursion (depth 1 >= 1)
                rlm = RLM(
                    model=self.model,
                    recursive_model=self.model, 
                    partition_strategy=None,
                    retrieval_method=None,
                    parallel_subqueries=False,
                    max_iterations=30,
                    max_depth=1, # Depth 1 means only root runs
                    temperature=1.0
                )
                answer = await rlm.acompletion(query=question, context=context)
                stats = rlm.stats
                llm_calls = stats['llm_calls']
                child_llm_calls = stats.get('child_llm_calls', 0)
                iterations = stats['iterations']
                depth = stats['depth']
                
            else: # mode == "rlm" (Standard Recursive)
                rlm = RLM(
                    model=self.model,
                    recursive_model=self.recursive_model,
                    partition_strategy=partition_strategy,
                    retrieval_method=retrieval_method,
                    parallel_subqueries=parallel_subqueries,
                    max_iterations=30,
                    max_depth=5,
                    temperature=1.0
                )
                answer = await rlm.acompletion(query=question, context=context)
                stats = rlm.stats
                llm_calls = stats['llm_calls']
                child_llm_calls = stats.get('child_llm_calls', 0)
                iterations = stats['iterations']
                depth = stats['depth']
            
            elapsed_time = time.time() - start_time
            
            # Calculate scores
            f1_score = self.calculate_f1(answer, ground_truth)
            oolong_score = self.calculate_oolong_score(answer, ground_truth, answer_type)
            exact_match = answer.strip().lower() == ground_truth.strip().lower() if ground_truth else False
            numeric_error = self.calculate_numeric_error(answer, ground_truth)

            # Log summary
            total_calls = llm_calls + child_llm_calls
            print(f"  -> Answer: {answer[:50]}{'...' if len(answer) > 50 else ''}")
            
            log_msg = f"  -> F1: {f1_score:.3f}, OOLONG Score: {oolong_score:.3f}, EM: {exact_match}"
            if numeric_error >= 0:
                log_msg += f", Num Error: {numeric_error:.1%}"
            log_msg += f", Time: {elapsed_time:.1f}s"
            print(log_msg)
            
            print(f"  -> LLM calls: {llm_calls} root + {child_llm_calls} child = {total_calls} total")
            
            return {
                'success': True,
                'answer': answer,
                'ground_truth': ground_truth,
                'f1_score': f1_score,
                'oolong_score': oolong_score,
                'exact_match': exact_match,
                'numeric_error': numeric_error,
                'context_length': len(context),
                'question': question,
                'llm_calls': llm_calls,
                'child_llm_calls': child_llm_calls,
                'total_llm_calls': total_calls,
                'iterations': iterations,
                'depth': depth,
                'elapsed_time': elapsed_time,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries,
                'mode': mode,
                'model': self.model
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'partition_strategy': partition_strategy,
                'retrieval_method': retrieval_method,
                'parallel_subqueries': parallel_subqueries,
                'mode': mode,
                'model': self.model,
                'f1_score': 0.0,
                'oolong_score': 0.0,
                'exact_match': False
            }
    
    async def evaluate_dataset(
        self,
        dataset,
        configs: List[Dict[str, Any]],
        max_examples: int = None
    ):
        """
        Evaluate on OOLONGBench with different configurations.
        
        Args:
            dataset: OOLONGBench dataset
            configs: List of configuration dictionaries
            max_examples: Maximum number of examples to evaluate
        """
        import random

        # Get examples from dataset
        if hasattr(dataset, 'keys'):
            examples = list(dataset['test']) if 'test' in dataset else list(dataset['train'])
        else:
            examples = list(dataset)
        
        # Pick random examples if max_examples is set
        if max_examples and len(examples) > max_examples:
            print(f"Selecting {max_examples} random examples from {len(examples)} total...")
            random.seed(42)  # Fixed seed for reproducibility across configurations
            examples = random.sample(examples, max_examples)
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {len(examples)} examples (Same examples for all configs)")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for config in configs:
            config_name = config['name']
            mode = config.get('mode', 'rlm')
            partition_strategy = config.get('partition_strategy')
            retrieval_method = config.get('retrieval_method')
            parallel = config.get('parallel_subqueries', False)
            model_override = config.get('model')
            
            # Update evaluator model if overridden for this config
            original_model = self.model
            if model_override:
                self.model = model_override
                
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name}")
            print(f"Mode: {mode}, Model: {self.model}")
            if partition_strategy:
                print(f"Partition: {partition_strategy}, Retrieval: {retrieval_method}")
            print(f"{'='*60}")
            
            config_results = []
            
            for i, example in enumerate(examples, 1):
                print(f"\nExample {i}/{len(examples)}")
                
                result = await self.evaluate_single_example(
                    example,
                    partition_strategy,
                    retrieval_method,
                    parallel,
                    mode
                )
                
                config_results.append(result)
                
                if not result['success']:
                    print(f"  [FAILED] {result['error']}")
                print()
                
                # Sleep to prevent rate limits
                print("Waiting 60s to respect rate limits...")
                await asyncio.sleep(60)
            
            # Save configuration results
            self.save_results(config_name, config_results)
            all_results.extend(config_results)
            
            # Restore model
            self.model = original_model
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def save_results(self, config_name: str, results: List[Dict[str, Any]]):
        """Save results to JSON file."""
        output_file = self.output_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Saved results to {output_file}")
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate and save summary statistics."""
        # Group results by configuration name
        configs = {}
        for result in all_results:
            if not result['success']:
                continue
            
            # Reconstruct the config name used in the loop
            # Or better, we should have stored config_name in result.
            # But we can infer it or just group by the unique params again.
            
            mode = result.get('mode')
            model = result.get('model')
            partition = result.get('partition_strategy')
            retrieval = result.get('retrieval_method')
            
            if mode == 'direct':
                name = f"{model}_Direct"
            elif mode == 'rlm-no-recursion':
                name = f"RLM_{model}_NoRecursion"
            else: # rlm
                if partition:
                    name = f"RLM_{model}_{partition}_{retrieval}"
                else:
                    name = f"RLM_{model}_REPL_Default"

            if name not in configs:
                configs[name] = {
                    'results': [],
                    'total_time': 0,
                    'total_llm_calls': 0,
                    'total_iterations': 0
                }
            
            configs[name]['results'].append(result)
            configs[name]['total_time'] += result['elapsed_time']
            configs[name]['total_llm_calls'] += result['total_llm_calls']
            configs[name]['total_iterations'] += result.get('iterations', 0)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configurations': {}
        }
        
        for name, data in configs.items():
            num_examples = len(data['results'])
            
            # Add OOLONG score aggregation
            oolong_scores = [r.get('oolong_score', 0.0) for r in data['results']]
            avg_oolong_score = sum(oolong_scores) / num_examples if num_examples > 0 else 0.0
            
            f1_scores = [r.get('f1_score', 0.0) for r in data['results']]
            avg_f1 = sum(f1_scores) / num_examples if num_examples > 0 else 0.0
            
            exact_matches = sum(1 for r in data['results'] if r.get('exact_match', False))
            
            summary['configurations'][name] = {
                'num_examples': num_examples,
                'avg_oolong_score': avg_oolong_score,
                'avg_f1_score': avg_f1,
                'exact_match_count': exact_matches,
                'exact_match_rate': exact_matches / num_examples if num_examples > 0 else 0.0,
                'avg_time': data['total_time'] / num_examples if num_examples > 0 else 0.0,
                'avg_llm_calls': data['total_llm_calls'] / num_examples if num_examples > 0 else 0.0,
                'avg_iterations': data['total_iterations'] / num_examples if num_examples > 0 else 0.0,
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
            print(f"  Avg OOLONG Score: {metrics['avg_oolong_score']:.3f}")
            print(f"  Avg Time: {metrics['avg_time']:.2f}s")
            print(f"  Avg LLM Calls: {metrics['avg_llm_calls']:.1f}")
            print()
        
        print(f"[OK] Saved summary to {summary_file}")


async def main():
    """Main evaluation function."""
    
    # Initialize evaluator with GPT-5-mini
    evaluator = OOLONGBenchEvaluator(
        model="gpt-5-mini",
        recursive_model="gpt-5-mini",
        output_dir="oolongbench_results"
    )
    
    # Load OOLONGBench dataset
    # We attempt 'trec_coarse' as requested, with fallback logic in the class
    dataset = evaluator.load_oolongbench(config="trec_coarse")
    
    # Define configurations to test based on Alex Zhang's evaluation
    configs = [
        # 2. GPT-5-mini (Direct)
        {
            "name": "GPT-5-mini_Direct",
            "mode": "direct",
            "model": "gpt-5-mini"
        },
        # 3. RLM(GPT-5-mini) - REPL Default (No explicit partition strategy)
        {
            "name": "RLM_GPT-5-mini_REPL_Default",
            "mode": "rlm",
            "model": "gpt-5-mini",
            "partition_strategy": None,
            "retrieval_method": None
        },
    ]

    # 5. RLM(GPT-5-mini) Partition Strategies
    # "Test the partition strategies"
    partition_strategies = ["token", "structural", "semantic"]
    retrieval_methods = ["unfiltered", "regex", "embedding"] 
    
    for strategy in partition_strategies:
        for retrieval in retrieval_methods:
             configs.append({
                "name": f"RLM_GPT-5-mini_{strategy}_{retrieval}",
                "mode": "rlm",
                "model": "gpt-5-mini",
                "partition_strategy": strategy,
                "retrieval_method": retrieval,
                "parallel_subqueries": False # Default sequential
             })

    # Run evaluation
    results = await evaluator.evaluate_dataset(
        dataset,
        configs,
        max_examples=20  # Run all examples in the loaded dataset
    )
    
    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found! Please set it in .env")
        exit(1)
        
    asyncio.run(main())
