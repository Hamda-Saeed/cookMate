"""
Analyze RAG evaluation results
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import json

# Load CSV results
df = pd.read_csv('rag_evaluation_results.csv')

# Separate RAG and Baseline
rag = df[df['model'] == 'RAG']
baseline = df[df['model'] == 'Baseline']

print("="*70)
print("RAG EVALUATION RESULTS ANALYSIS")
print("="*70)

print("\n=== RAG MODEL PERFORMANCE ===")
print(f"Exact Match Rate:        {rag['exact_match'].mean()*100:.2f}%")
print(f"Avg Partial Match (F1):  {rag['partial_match'].mean():.4f}")
print(f"Avg BLEU Score:          {rag['bleu'].mean():.4f}")
print(f"Avg ROUGE-1:             {rag['rouge1'].mean():.4f}")
print(f"Avg ROUGE-2:             {rag['rouge2'].mean():.4f}")
print(f"Avg ROUGE-L:             {rag['rougeL'].mean():.4f}")
print(f"Avg Semantic Similarity: {rag['semantic_similarity'].mean():.4f} ({rag['semantic_similarity'].mean()*100:.2f}%)")
print(f"Avg Latency:              {rag['latency'].mean():.3f}s")
print(f"Zero Latency Count:      {(rag['latency']==0).sum()}/{len(rag)} questions ({((rag['latency']==0).sum()/len(rag)*100):.1f}%)")

print("\n=== BASELINE MODEL PERFORMANCE ===")
print(f"Exact Match Rate:        {baseline['exact_match'].mean()*100:.2f}%")
print(f"Avg Partial Match (F1):  {baseline['partial_match'].mean():.4f}")
print(f"Avg BLEU Score:          {baseline['bleu'].mean():.4f}")
print(f"Avg ROUGE-1:             {baseline['rouge1'].mean():.4f}")
print(f"Avg ROUGE-2:             {baseline['rouge2'].mean():.4f}")
print(f"Avg ROUGE-L:             {baseline['rougeL'].mean():.4f}")
print(f"Avg Semantic Similarity: {baseline['semantic_similarity'].mean():.4f} ({baseline['semantic_similarity'].mean()*100:.2f}%)")
print(f"Avg Latency:              {baseline['latency'].mean():.3f}s")
print(f"Zero Latency Count:      {(baseline['latency']==0).sum()}/{len(baseline)} questions ({((baseline['latency']==0).sum()/len(baseline)*100):.1f}%)")

print("\n=== IMPROVEMENTS (RAG vs Baseline) ===")
improvements = {
    'Partial Match': ((rag['partial_match'].mean() - baseline['partial_match'].mean()) / baseline['partial_match'].mean() * 100) if baseline['partial_match'].mean() > 0 else 0,
    'BLEU': ((rag['bleu'].mean() - baseline['bleu'].mean()) / baseline['bleu'].mean() * 100) if baseline['bleu'].mean() > 0 else 0,
    'ROUGE-1': ((rag['rouge1'].mean() - baseline['rouge1'].mean()) / baseline['rouge1'].mean() * 100) if baseline['rouge1'].mean() > 0 else 0,
    'ROUGE-2': ((rag['rouge2'].mean() - baseline['rouge2'].mean()) / baseline['rouge2'].mean() * 100) if baseline['rouge2'].mean() > 0 else 0,
    'ROUGE-L': ((rag['rougeL'].mean() - baseline['rougeL'].mean()) / baseline['rougeL'].mean() * 100) if baseline['rougeL'].mean() > 0 else 0,
    'Semantic Similarity': ((rag['semantic_similarity'].mean() - baseline['semantic_similarity'].mean()) / baseline['semantic_similarity'].mean() * 100) if baseline['semantic_similarity'].mean() > 0 else 0,
}

for metric, improvement in improvements.items():
    print(f"{metric:<20}: {improvement:+.1f}%")

print("\n=== ISSUES IDENTIFIED ===")

# Check for zero latency (fallback responses)
rag_zero_latency = (rag['latency'] == 0).sum()
baseline_zero_latency = (baseline['latency'] == 0).sum()

if rag_zero_latency > 0:
    print(f"WARNING: RAG Model: {rag_zero_latency} questions with 0 latency (likely fallback responses)")
    print(f"   This suggests rate limiting or API errors occurred")
    print(f"   CRITICAL: This is causing low scores!")

if baseline_zero_latency > 0:
    print(f"WARNING: Baseline Model: {baseline_zero_latency} questions with 0 latency (likely fallback responses)")

# Check for negative semantic similarity (shouldn't happen)
rag_negative = (rag['semantic_similarity'] < 0).sum()
baseline_negative = (baseline['semantic_similarity'] < 0).sum()

if rag_negative > 0:
    print(f"WARNING: RAG Model: {rag_negative} questions with negative semantic similarity (data issue)")

if baseline_negative > 0:
    print(f"WARNING: Baseline Model: {baseline_negative} questions with negative semantic similarity (data issue)")

# Check for very low scores
rag_low_sim = (rag['semantic_similarity'] < 0.2).sum()
baseline_low_sim = (baseline['semantic_similarity'] < 0.2).sum()

print(f"\n📊 Questions with low semantic similarity (<0.2):")
print(f"   RAG: {rag_low_sim}/100 ({rag_low_sim}%)")
print(f"   Baseline: {baseline_low_sim}/100 ({baseline_low_sim}%)")

# Best and worst performing questions for testing
print("\n=== BEST PERFORMING QUESTIONS (RAG) ===")
top_rag = rag.nlargest(5, 'semantic_similarity')[['question_id', 'semantic_similarity', 'rougeL', 'latency']]
for idx, row in top_rag.iterrows():
    print(f"Q{int(row['question_id']):3d}: Sim={row['semantic_similarity']:.3f}, ROUGE-L={row['rougeL']:.3f}, Latency={row['latency']:.2f}s")

print("\n=== WORST PERFORMING QUESTIONS (RAG) ===")
worst_rag = rag.nsmallest(5, 'semantic_similarity')[['question_id', 'semantic_similarity', 'rougeL', 'latency']]
for idx, row in worst_rag.iterrows():
    print(f"Q{int(row['question_id']):3d}: Sim={row['semantic_similarity']:.3f}, ROUGE-L={row['rougeL']:.3f}, Latency={row['latency']:.2f}s")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

