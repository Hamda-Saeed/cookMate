"""
Main script to run RAG evaluation
Compares RAG-based model vs Baseline model on 100 QA pairs
"""

import json
import sys
import os
from datetime import datetime

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import required modules
from cookmate_rag import CookMateRAG
from baseline_model import BaselineModel
from rag_evaluation import RAGEvaluator
from generate_qa_dataset import generate_qa_pairs, save_dataset

def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_section(text: str):
    """Print formatted section header"""
    print(f"\n{'-'*70}")
    print(f"  {text}")
    print(f"{'-'*70}\n")

def main():
    """Main evaluation pipeline"""
    
    print_header("CookMate RAG Evaluation System")
    print("This script will:")
    print("  1. Generate 100 QA pairs from recipes.json")
    print("  2. Evaluate RAG-based model")
    print("  3. Evaluate Baseline model (without RAG)")
    print("  4. Compare performance and generate report")
    print()
    
    # Step 1: Generate or load QA dataset
    dataset_path = "rag_qa_dataset.json"
    
    if not os.path.exists(dataset_path):
        print_section("Step 1: Generating QA Dataset")
        print("Generating 100 Question-Answer pairs from recipes.json...")
        qa_pairs = generate_qa_pairs("recipes.json", num_pairs=100)
        save_dataset(qa_pairs, dataset_path)
    else:
        print_section("Step 1: Loading QA Dataset")
        print(f"Loading existing dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            qa_pairs = dataset['qa_pairs']
        print(f"Loaded {len(qa_pairs)} QA pairs")
        print(f"   Document source: {dataset['metadata']['document_source']}")
        print(f"   Question types: {dataset['metadata']['question_types']}")
    
    # Step 2: Initialize models
    print_section("Step 2: Initializing Models")
    
    print("Initializing RAG Model...")
    try:
        rag_model = CookMateRAG(
            recipe_data_path="recipes.json",
            use_whisper=False  # Disable Whisper for faster evaluation
        )
        print("RAG Model ready!")
    except Exception as e:
        print(f"Error initializing RAG model: {e}")
        print("   Make sure you have set up your Groq API key in cookmate_rag.py")
        return
    
    print("\nInitializing Baseline Model...")
    try:
        baseline_model = BaselineModel()
        print("Baseline Model ready!")
    except Exception as e:
        print(f"Error initializing Baseline model: {e}")
        return
    
    # Step 3: Initialize evaluator
    print_section("Step 3: Initializing Evaluator")
    evaluator = RAGEvaluator()
    
    # Step 4: Evaluate RAG model to get bleu rouge scores exact match partial match
    print_section("Step 4: Evaluating RAG Model")
    print("Running RAG model on all questions...")
    print("   This may take several minutes...\n")
    
    try:
        rag_results = evaluator.evaluate_model(
            rag_model,
            qa_pairs,
            model_name="RAG Model"
        )
    except Exception as e:
        print(f"Error evaluating RAG model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Evaluate Baseline model
    print_section("Step 5: Evaluating Baseline Model")
    print("Running Baseline model on all questions...")
    print("   This may take several minutes...\n")
    
    try:
        baseline_results = evaluator.evaluate_model(
            baseline_model,
            qa_pairs,
            model_name="Baseline Model"
        )
    except Exception as e:
        print(f"Error evaluating Baseline model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Compare models
    print_section("Step 6: Comparing Models")
    print("Comparing RAG vs Baseline performance...\n")
    
    comparison = evaluator.compare_models(rag_results, baseline_results)
    
    # Print comparison summary
    print("Performance Comparison Summary:")
    print(f"\n{'Metric':<30} {'RAG':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-" * 75)
    
    improvements = comparison['improvements']
    for metric, data in improvements.items():
        if metric != 'avg_latency':
            rag_val = data['rag']
            baseline_val = data['baseline']
            improvement = data['improvement_percent']
            
            if isinstance(rag_val, float):
                rag_str = f"{rag_val:.4f}"
                baseline_str = f"{baseline_val:.4f}"
                improvement_str = f"{improvement:+.2f}%"
            else:
                rag_str = str(rag_val)
                baseline_str = str(baseline_val)
                improvement_str = f"{improvement:+.2f}%"
            
            print(f"{metric:<30} {rag_str:<15} {baseline_str:<15} {improvement_str:<15}")
    
    # Latency comparison
    latency_data = improvements['avg_latency']
    print(f"\n{'avg_latency':<30} {latency_data['rag']:.3f}s{'':<10} {latency_data['baseline']:.3f}s{'':<10} {latency_data['difference']:+.3f}s")
    
    # Step 7: Generate report
    print_section("Step 7: Generating Report")
    
    report = evaluator.generate_report(
        rag_results,
        baseline_results,
        comparison,
        output_path="rag_evaluation_report.json"
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"   Total Questions: {report['metadata']['total_questions']}")
    print(f"   Evaluation Date: {report['metadata']['evaluation_date']}")
    
    print("\nKey Findings:")
    for finding in report['summary']['key_findings']:
        print(f"   - {finding}")
    
    print(f"\nBest Performing Model: {report['summary']['best_performing_model']}")
    
    # Export to CSV for further analysis
    try:
        df = evaluator.export_to_dataframe(rag_results, baseline_results)
        csv_path = "rag_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results exported to: {csv_path}")
    except Exception as e:
        print(f"\nWarning: Could not export to CSV: {e}")
    
    print_header("Evaluation Complete!")
    print("Full report saved to: rag_evaluation_report.json")
    print("Detailed results saved to: rag_evaluation_results.csv")
    print("\nYou can now analyze the results to compare RAG vs Baseline performance!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

