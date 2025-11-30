"""
Comprehensive RAG Evaluation System
Compares RAG-based model vs Baseline model (without RAG)
"""

import json
import time
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import pandas as pd

# Evaluation metrics
import warnings
warnings.filterwarnings('ignore')

# Try to import required packages with error handling
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk_available = True
except ImportError:
    print("Warning: nltk not available. BLEU scores will be disabled.")
    nltk_available = False
    sentence_bleu = None
    SmoothingFunction = None

try:
    from rouge_score import rouge_scorer
    rouge_available = True
except ImportError:
    print("Warning: rouge-score not available. ROUGE scores will be disabled.")
    print("Install with: pip install rouge-score")
    rouge_available = False
    rouge_scorer = None

try:
    from sentence_transformers import SentenceTransformer
    st_available = True
except ImportError:
    print("Warning: sentence-transformers not available. Semantic similarity will be disabled.")
    st_available = False
    SentenceTransformer = None

# Try to import nltk and download required data
if nltk_available:
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except:
        pass


class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG vs Baseline comparison
    """
    
    def __init__(self):
        """Initialize evaluator with metrics"""
        print("Initializing RAG Evaluator...")
        
        # Load semantic similarity model
        if st_available:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.similarity_model = None
        else:
            self.similarity_model = None
        
        # Initialize ROUGE scorer
        if rouge_available:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            except Exception as e:
                print(f"Warning: Could not initialize ROUGE scorer: {e}")
                self.rouge_scorer = None
        else:
            self.rouge_scorer = None
        
        # Smoothing function for BLEU
        if nltk_available and SmoothingFunction:
            self.smoothing = SmoothingFunction().method1
        else:
            self.smoothing = None
        
        print("Evaluator ready!\n")
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer exactly matches ground truth (case-insensitive)
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            True if exact match, False otherwise
        """
        return predicted.strip().lower() == ground_truth.strip().lower()
    
    def partial_match(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate partial match score based on word overlap
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            F1 score of word overlap
        """
        pred_words = set(predicted.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if len(gt_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        if len(pred_words) == 0:
            return 0.0
        
        intersection = pred_words & gt_words
        precision = len(intersection) / len(pred_words) if len(pred_words) > 0 else 0.0
        recall = len(intersection) / len(gt_words) if len(gt_words) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def bleu_score(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate BLEU score between predicted and ground truth
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            BLEU score (0-1)
        """
        if not nltk_available or not sentence_bleu or not self.smoothing:
            return 0.0
        
        try:
            reference = [ground_truth.lower().split()]
            candidate = predicted.lower().split()
            
            score = sentence_bleu(
                reference,
                candidate,
                smoothing_function=self.smoothing
            )
            return score
        except:
            return 0.0
    
    def rouge_scores(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """
        Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            Dictionary with rouge1, rouge2, rougeL scores
        """
        if not rouge_available or not self.rouge_scorer:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(ground_truth, predicted)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate cosine similarity between embeddings of predicted and ground truth
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not st_available or not self.similarity_model:
            return 0.0
        
        try:
            pred_embedding = self.similarity_model.encode(predicted, convert_to_numpy=True)
            gt_embedding = self.similarity_model.encode(ground_truth, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = np.dot(pred_embedding, gt_embedding) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(gt_embedding)
            )
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> Dict:
        """
        Evaluate a single answer using all metrics
        
        Args:
            predicted: Model-generated answer
            ground_truth: Correct answer
            
        Returns:
            Dictionary with all metric scores
        """
        # Calculate all metrics
        exact = self.exact_match(predicted, ground_truth)
        partial = self.partial_match(predicted, ground_truth)
        bleu = self.bleu_score(predicted, ground_truth)
        rouge = self.rouge_scores(predicted, ground_truth)
        semantic = self.semantic_similarity(predicted, ground_truth)
        
        return {
            'exact_match': exact,
            'partial_match': partial,
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'semantic_similarity': semantic
        }
    
    def evaluate_model(self, 
                      model, 
                      qa_pairs: List[Dict],
                      model_name: str = "Model") -> Dict:
        """
        Evaluate a model on the entire dataset
        
        Args:
            model: Model object with generate_answer method
            qa_pairs: List of QA pairs from dataset
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with aggregated metrics and per-question results
        """
        print(f"\nEvaluating {model_name}...")
        print(f"   Processing {len(qa_pairs)} questions...")
        
        results = []
        latencies = []
        
        for i, qa in enumerate(qa_pairs, 1):
            question = qa['question']
            ground_truth = qa['answer']
            
            # Generate answer and measure latency
            start_time = time.time()
            try:
                if hasattr(model, 'process_query'):
                    # RAG model
                    predicted, latency = model.process_query(question)
                else:
                    # Baseline model
                    predicted = model.generate_answer(question)
                    latency = time.time() - start_time
                
                # Small delay to help with rate limiting (rate limiting is handled in GroqLLM)
                if i < len(qa_pairs):  # Don't delay after last question
                    time.sleep(0.05)  # Small delay
            except Exception as e:
                predicted = f"Error: {str(e)}"
                latency = time.time() - start_time
            
            latencies.append(latency)
            
            # Evaluate answer
            metrics = self.evaluate_answer(predicted, ground_truth)
            
            result = {
                'question_id': qa['id'],
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'latency': latency,
                **metrics
            }
            results.append(result)
            
            if i % 10 == 0:
                print(f"   Processed {i}/{len(qa_pairs)} questions...")
        
        # Aggregate metrics
        aggregated = {
            'model_name': model_name,
            'total_questions': len(qa_pairs),
            'exact_match_rate': np.mean([r['exact_match'] for r in results]),
            'avg_partial_match': np.mean([r['partial_match'] for r in results]),
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'avg_rouge1': np.mean([r['rouge1'] for r in results]),
            'avg_rouge2': np.mean([r['rouge2'] for r in results]),
            'avg_rougeL': np.mean([r['rougeL'] for r in results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in results]),
            'avg_latency': np.mean(latencies),
            'total_latency': sum(latencies)
        }
        
        print(f"{model_name} evaluation complete!")
        print(f"   Exact Match Rate: {aggregated['exact_match_rate']*100:.2f}%")
        print(f"   Avg Semantic Similarity: {aggregated['avg_semantic_similarity']*100:.2f}%")
        print(f"   Avg Latency: {aggregated['avg_latency']:.3f}s")
        
        return {
            'aggregated': aggregated,
            'detailed_results': results
        }
    
    def compare_models(self, 
                      rag_results: Dict, 
                      baseline_results: Dict) -> Dict:
        """
        Compare RAG vs Baseline model performance
        
        Args:
            rag_results: Results from RAG model evaluation
            baseline_results: Results from baseline model evaluation
            
        Returns:
            Comparison dictionary with improvements
        """
        rag_agg = rag_results['aggregated']
        baseline_agg = baseline_results['aggregated']
        
        comparison = {
            'rag_model': rag_agg,
            'baseline_model': baseline_agg,
            'improvements': {}
        }
        
        # Calculate improvements
        metrics_to_compare = [
            'exact_match_rate', 'avg_partial_match', 'avg_bleu',
            'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_semantic_similarity'
        ]
        
        for metric in metrics_to_compare:
            rag_val = rag_agg[metric]
            baseline_val = baseline_agg[metric]
            
            if baseline_val > 0:
                improvement = ((rag_val - baseline_val) / baseline_val) * 100
            else:
                improvement = (rag_val - baseline_val) * 100
            
            comparison['improvements'][metric] = {
                'rag': rag_val,
                'baseline': baseline_val,
                'improvement_percent': improvement,
                'absolute_improvement': rag_val - baseline_val
            }
        
        # Latency comparison
        comparison['improvements']['avg_latency'] = {
            'rag': rag_agg['avg_latency'],
            'baseline': baseline_agg['avg_latency'],
            'difference': rag_agg['avg_latency'] - baseline_agg['avg_latency'],
            'difference_percent': ((rag_agg['avg_latency'] - baseline_agg['avg_latency']) / baseline_agg['avg_latency']) * 100 if baseline_agg['avg_latency'] > 0 else 0
        }
        
        return comparison
    
    def generate_report(self, 
                       rag_results: Dict,
                       baseline_results: Dict,
                       comparison: Dict,
                       output_path: str = "rag_evaluation_report.json") -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            rag_results: RAG model evaluation results
            baseline_results: Baseline model evaluation results
            comparison: Comparison dictionary
            output_path: Path to save report
            
        Returns:
            Complete report dictionary
        """
        report = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_questions': rag_results['aggregated']['total_questions'],
                'evaluation_metrics': [
                    'Exact Match', 'Partial Match (F1)', 'BLEU', 
                    'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Similarity'
                ]
            },
            'rag_model_results': rag_results['aggregated'],
            'baseline_model_results': baseline_results['aggregated'],
            'comparison': comparison,
            'summary': self._generate_summary(comparison)
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {output_path}")
        
        return report
    
    def _generate_summary(self, comparison: Dict) -> Dict:
        """Generate text summary of comparison"""
        improvements = comparison['improvements']
        
        summary = {
            'key_findings': [],
            'best_performing_model': 'RAG' if improvements['avg_semantic_similarity']['improvement_percent'] > 0 else 'Baseline',
            'metric_improvements': {}
        }
        
        # Key findings
        if improvements['exact_match_rate']['improvement_percent'] > 0:
            summary['key_findings'].append(
                f"RAG model shows {improvements['exact_match_rate']['improvement_percent']:.1f}% improvement in exact match rate"
            )
        
        if improvements['avg_semantic_similarity']['improvement_percent'] > 0:
            summary['key_findings'].append(
                f"RAG model achieves {improvements['avg_semantic_similarity']['improvement_percent']:.1f}% better semantic similarity"
            )
        
        if improvements['avg_rougeL']['improvement_percent'] > 0:
            summary['key_findings'].append(
                f"RAG model shows {improvements['avg_rougeL']['improvement_percent']:.1f}% improvement in ROUGE-L score"
            )
        
        # Metric improvements
        for metric, data in improvements.items():
            if metric != 'avg_latency':
                summary['metric_improvements'][metric] = {
                    'improvement': f"{data['improvement_percent']:.2f}%",
                    'rag_score': f"{data['rag']:.4f}",
                    'baseline_score': f"{data['baseline']:.4f}"
                }
        
        return summary
    
    def export_to_dataframe(self, rag_results: Dict, baseline_results: Dict) -> pd.DataFrame:
        """Export results to pandas DataFrame for analysis"""
        all_results = []
        
        # Add RAG results
        for result in rag_results['detailed_results']:
            all_results.append({
                'model': 'RAG',
                'question_id': result['question_id'],
                'exact_match': result['exact_match'],
                'partial_match': result['partial_match'],
                'bleu': result['bleu'],
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL'],
                'semantic_similarity': result['semantic_similarity'],
                'latency': result['latency']
            })
        
        # Add baseline results
        for result in baseline_results['detailed_results']:
            all_results.append({
                'model': 'Baseline',
                'question_id': result['question_id'],
                'exact_match': result['exact_match'],
                'partial_match': result['partial_match'],
                'bleu': result['bleu'],
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL'],
                'semantic_similarity': result['semantic_similarity'],
                'latency': result['latency']
            })
        
        return pd.DataFrame(all_results)

