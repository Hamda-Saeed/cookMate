"""
CookMate Evaluation Module
Implements all metrics from the project proposal
"""

import json
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from jiwer import wer
import pandas as pd


class CookMateEvaluator:
    """Evaluation metrics for CookMate system"""
    
    def __init__(self):
        self.test_results = []
        self.wer_scores = []
        self.retrieval_scores = []
        self.latencies = []
        self.relevance_scores = []
    
    def evaluate_step_retrieval(self, 
                                ground_truth_step: int, 
                                retrieved_step: int,
                                recipe_name: str = "") -> Dict:
        """
        Step Retrieval Accuracy Metric
        Target: >95% accuracy
        """
        is_correct = (ground_truth_step == retrieved_step)
        result = {
            "recipe": recipe_name,
            "ground_truth": ground_truth_step,
            "retrieved": retrieved_step,
            "correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        self.retrieval_scores.append(result)
        return result
    
    def evaluate_asr_wer(self, 
                        reference_text: str, 
                        hypothesis_text: str) -> Dict:
        """
        Word Error Rate (WER) for ASR
        Target: <10% WER
        """
        wer_score = wer(reference_text, hypothesis_text)
        
        result = {
            "reference": reference_text,
            "hypothesis": hypothesis_text,
            "wer": wer_score,
            "meets_target": wer_score < 0.10,
            "timestamp": datetime.now().isoformat()
        }
        
        self.wer_scores.append(result)
        return result
    
    def evaluate_response_relevance(self, 
                                    query: str,
                                    response: str,
                                    human_rating: int) -> Dict:
        """
        Response Relevance & Correctness
        Human evaluation on 1-5 Likert scale
        """
        if not (1 <= human_rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        result = {
            "query": query,
            "response": response,
            "rating": human_rating,
            "timestamp": datetime.now().isoformat()
        }
        
        self.relevance_scores.append(result)
        return result
    
    def evaluate_latency(self, 
                        latency_seconds: float,
                        operation_type: str = "e2e") -> Dict:
        """
        End-to-End Latency
        Target: <3 seconds from speech input to TTS output
        """
        meets_target = latency_seconds < 3.0
        
        result = {
            "latency": latency_seconds,
            "operation": operation_type,
            "meets_target": meets_target,
            "timestamp": datetime.now().isoformat()
        }
        
        self.latencies.append(result)
        return result
    
    def calculate_sus_score(self, responses: List[int]) -> Dict:
        """
        System Usability Scale (SUS)
        10 questions, each rated 1-5
        """
        if len(responses) != 10:
            raise ValueError("SUS requires exactly 10 responses")
        
        if not all(1 <= r <= 5 for r in responses):
            raise ValueError("All responses must be between 1 and 5")
        
        # Calculate SUS score
        # Odd questions (1,3,5,7,9): subtract 1
        # Even questions (2,4,6,8,10): subtract from 5
        score = 0
        for i, response in enumerate(responses):
            if i % 2 == 0:  # Odd questions (0-indexed)
                score += response - 1
            else:  # Even questions
                score += 5 - response
        
        # Multiply by 2.5 to get score out of 100
        sus_score = score * 2.5
        
        # Interpret score
        if sus_score >= 80:
            grade = "A - Excellent"
        elif sus_score >= 70:
            grade = "B - Good"
        elif sus_score >= 50:
            grade = "C - OK"
        elif sus_score >= 25:
            grade = "D - Poor"
        else:
            grade = "F - Awful"
        
        return {
            "sus_score": sus_score,
            "grade": grade,
            "responses": responses,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_overall_metrics(self) -> Dict:
        """Calculate overall system performance metrics"""
        metrics = {}
        
        # Step Retrieval Accuracy
        if self.retrieval_scores:
            total_correct = sum(s['correct'] for s in self.retrieval_scores)
            accuracy = (total_correct / len(self.retrieval_scores)) * 100
            metrics['step_retrieval'] = {
                "accuracy": f"{accuracy:.2f}%",
                "total_tests": len(self.retrieval_scores),
                "correct": total_correct,
                "meets_target": accuracy > 95.0
            }
        
        # WER
        if self.wer_scores:
            avg_wer = np.mean([s['wer'] for s in self.wer_scores])
            metrics['asr_wer'] = {
                "average_wer": f"{avg_wer*100:.2f}%",
                "total_tests": len(self.wer_scores),
                "meets_target": avg_wer < 0.10
            }
        
        # Latency
        if self.latencies:
            avg_latency = np.mean([l['latency'] for l in self.latencies])
            max_latency = max([l['latency'] for l in self.latencies])
            min_latency = min([l['latency'] for l in self.latencies])
            within_target = sum(1 for l in self.latencies if l['meets_target'])
            
            metrics['latency'] = {
                "average": f"{avg_latency:.3f}s",
                "min": f"{min_latency:.3f}s",
                "max": f"{max_latency:.3f}s",
                "within_target": f"{(within_target/len(self.latencies))*100:.1f}%",
                "meets_target": avg_latency < 3.0
            }
        
        # Response Relevance
        if self.relevance_scores:
            avg_rating = np.mean([s['rating'] for s in self.relevance_scores])
            metrics['response_relevance'] = {
                "average_rating": f"{avg_rating:.2f}/5.0",
                "total_ratings": len(self.relevance_scores),
                "percentage": f"{(avg_rating/5.0)*100:.1f}%"
            }
        
        return metrics
    
    def generate_report(self, output_file: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": self.get_overall_metrics(),
            "detailed_results": {
                "step_retrieval": self.retrieval_scores,
                "asr_wer": self.wer_scores,
                "latency": self.latencies,
                "response_relevance": self.relevance_scores
            },
            "summary": self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate text summary of results"""
        metrics = self.get_overall_metrics()
        
        summary = {
            "total_tests_run": (
                len(self.retrieval_scores) + 
                len(self.wer_scores) + 
                len(self.latencies) + 
                len(self.relevance_scores)
            ),
            "targets_met": []
        }
        
        # Check which targets are met
        if metrics.get('step_retrieval', {}).get('meets_target'):
            summary['targets_met'].append("Step Retrieval Accuracy (>95%)")
        
        if metrics.get('asr_wer', {}).get('meets_target'):
            summary['targets_met'].append("ASR Word Error Rate (<10%)")
        
        if metrics.get('latency', {}).get('meets_target'):
            summary['targets_met'].append("End-to-End Latency (<3s)")
        
        return summary
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all results to pandas DataFrame for analysis"""
        all_results = []
        
        for score in self.retrieval_scores:
            all_results.append({
                'metric_type': 'step_retrieval',
                'value': score['accuracy'],
                'timestamp': score['timestamp']
            })
        
        for score in self.wer_scores:
            all_results.append({
                'metric_type': 'wer',
                'value': score['wer'],
                'timestamp': score['timestamp']
            })
        
        for latency in self.latencies:
            all_results.append({
                'metric_type': 'latency',
                'value': latency['latency'],
                'timestamp': latency['timestamp']
            })
        
        for relevance in self.relevance_scores:
            all_results.append({
                'metric_type': 'relevance',
                'value': relevance['rating'],
                'timestamp': relevance['timestamp']
            })
        
        return pd.DataFrame(all_results)


# Test suite for automated evaluation
class CookMateTestSuite:
    """Automated test cases for CookMate"""
    
    def __init__(self, cookmate_system):
        self.system = cookmate_system
        self.evaluator = CookMateEvaluator()
    
    def test_step_navigation(self) -> Dict:
        """Test step-by-step navigation accuracy"""
        print("\n🧪 Testing Step Navigation...")
        
        # Start a recipe
        self.system.process_query("Start pasta carbonara")
        
        results = []
        for i in range(1, 6):  # 5 steps
            response, _ = self.system.process_query("What's next?")
            
            # Check if correct step
            current_step = self.system.state.current_step
            result = self.evaluator.evaluate_step_retrieval(
                ground_truth_step=i,
                retrieved_step=current_step,
                recipe_name="Pasta Carbonara"
            )
            results.append(result)
            print(f"  Step {i}: {'✓' if result['correct'] else '✗'}")
        
        return results
    
    def test_asr_accuracy(self, test_pairs: List[Tuple[str, str]]) -> Dict:
        """Test ASR accuracy with reference-hypothesis pairs"""
        print("\n🧪 Testing ASR Accuracy...")
        
        results = []
        for reference, hypothesis in test_pairs:
            result = self.evaluator.evaluate_asr_wer(reference, hypothesis)
            results.append(result)
            status = "✓" if result['meets_target'] else "✗"
            print(f"  WER: {result['wer']*100:.2f}% {status}")
        
        return results
    
    def test_latency(self, num_queries: int = 10) -> Dict:
        """Test system latency"""
        print(f"\n🧪 Testing Latency ({num_queries} queries)...")
        
        test_queries = [
            "What's next?",
            "Repeat that",
            "How long does this take?",
            "What ingredients do I need?",
            "Can I substitute pancetta?",
        ]
        
        results = []
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            _, latency = self.system.process_query(query)
            
            result = self.evaluator.evaluate_latency(latency, "query_processing")
            results.append(result)
            status = "✓" if result['meets_target'] else "✗"
            print(f"  Query {i+1}: {latency:.3f}s {status}")
        
        return results
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite"""
        print("\n" + "="*60)
        print("🚀 CookMate Full Evaluation Suite")
        print("="*60)
        
        # Test 1: Step Navigation
        self.test_step_navigation()
        
        # Test 2: ASR (with sample data)
        sample_asr_tests = [
            ("what's next", "what's next"),
            ("repeat that step", "repeat that step"),
            ("how long does this take", "how long does it take"),
        ]
        self.test_asr_accuracy(sample_asr_tests)
        
        # Test 3: Latency
        self.test_latency(num_queries=10)
        
        # Generate report
        print("\n📊 Generating Evaluation Report...")
        report = self.evaluator.generate_report()
        
        print("\n" + "="*60)
        print("✅ Evaluation Complete!")
        print("="*60)
        
        # Print summary
        metrics = self.evaluator.get_overall_metrics()
        for metric_name, metric_data in metrics.items():
            print(f"\n{metric_name.upper()}:")
            for key, value in metric_data.items():
                print(f"  {key}: {value}")
        
        return report


# Example usage
if __name__ == "__main__":
    from cookmate_rag import CookMateRAG
    
    # Initialize system
    print("Initializing CookMate for evaluation...")
    cookmate = CookMateRAG(use_whisper=False)
    
    # Run test suite
    test_suite = CookMateTestSuite(cookmate)
    report = test_suite.run_full_evaluation()
    
    print("\n📄 Report saved to: evaluation_report.json")