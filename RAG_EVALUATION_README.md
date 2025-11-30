# RAG Evaluation System Documentation

This document explains how to use the RAG evaluation system to compare RAG-based models vs baseline models.

## Overview

The evaluation system includes:
1. **Dataset Generator** (`generate_qa_dataset.py`) - Creates 100 Question-Answer pairs from `recipes.json`
2. **Baseline Model** (`baseline_model.py`) - LLM without RAG access
3. **RAG Evaluator** (`rag_evaluation.py`) - Comprehensive evaluation metrics
4. **Evaluation Runner** (`run_rag_evaluation.py`) - Main script to run full evaluation

## Dataset Structure

The generated dataset (`rag_qa_dataset.json`) contains:
- **100 QA pairs** covering:
  - Recipe overview questions (name, prep time, cook time, servings, difficulty)
  - Ingredient questions (list, quantities, specific ingredients)
  - Step-by-step questions (instructions, durations, tips)
  - Multi-recipe comparison questions
  - Timing questions
  - Cooking technique questions

- Each QA pair includes:
  - `question`: The question text
  - `answer`: Ground truth answer
  - `document`: Source document name (`recipes.json`)
  - `pages`: Page/section reference (e.g., "Page 1 (Recipe Overview Section)")
  - `recipe_id`: Recipe identifier
  - `recipe_name`: Recipe name
  - `question_type`: Type of question

## Evaluation Metrics

The system evaluates models using:

1. **Exact Match**: Binary score (1 if exact match, 0 otherwise)
2. **Partial Match (F1)**: Word overlap F1 score
3. **BLEU Score**: Bilingual Evaluation Understudy score
4. **ROUGE Scores**: 
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence
5. **Semantic Similarity**: Cosine similarity between embeddings
6. **Latency**: Response time per question

## Usage

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required additional packages:
- `nltk` - For BLEU score calculation
- `rouge-score` - For ROUGE metrics

### Step 2: Generate QA Dataset

```bash
python generate_qa_dataset.py
```

This creates `rag_qa_dataset.json` with 100 QA pairs. You can specify a different number:

```bash
python generate_qa_dataset.py 50  # Generate 50 QA pairs
```

### Step 3: Run Evaluation

```bash
python run_rag_evaluation.py
```

This script will:
1. Load or generate the QA dataset
2. Initialize both RAG and Baseline models
3. Evaluate both models on all questions
4. Compare performance
5. Generate comprehensive report

### Step 4: Review Results

After evaluation, you'll get:

1. **`rag_evaluation_report.json`** - Complete evaluation report with:
   - Aggregated metrics for both models
   - Per-question detailed results
   - Comparison and improvements
   - Summary findings

2. **`rag_evaluation_results.csv`** - Detailed results in CSV format for analysis

## Understanding the Results

### Report Structure

```json
{
  "metadata": {
    "evaluation_date": "...",
    "total_questions": 100,
    "evaluation_metrics": [...]
  },
  "rag_model_results": {
    "exact_match_rate": 0.XX,
    "avg_semantic_similarity": 0.XX,
    ...
  },
  "baseline_model_results": {
    ...
  },
  "comparison": {
    "improvements": {
      "exact_match_rate": {
        "rag": 0.XX,
        "baseline": 0.XX,
        "improvement_percent": XX.XX
      },
      ...
    }
  },
  "summary": {
    "key_findings": [...],
    "best_performing_model": "RAG" or "Baseline"
  }
}
```

### Key Metrics to Look For

1. **Exact Match Rate**: Percentage of answers that exactly match ground truth
2. **Semantic Similarity**: How semantically similar answers are (0-1 scale)
3. **ROUGE-L**: Measures answer quality and completeness
4. **Latency**: Response time (RAG may be slightly slower due to retrieval)

### Expected Results

- **RAG Model** should show:
  - Higher exact match rate (access to recipe data)
  - Better semantic similarity (context-aware answers)
  - More accurate ingredient/step information
  
- **Baseline Model** may:
  - Have lower accuracy (no access to specific recipes)
  - Show general knowledge but miss specific details
  - Potentially faster (no retrieval step)

## Customization

### Adding More Question Types

Edit `generate_qa_dataset.py` to add new question types in the `generate_qa_pairs()` function.

### Changing Evaluation Metrics

Modify `rag_evaluation.py` to add or remove metrics in the `evaluate_answer()` method.

### Using Different Models

1. **RAG Model**: Modify `CookMateRAG` initialization in `run_rag_evaluation.py`
2. **Baseline Model**: Modify `BaselineModel` initialization or create a custom baseline

## Troubleshooting

### Issue: UnicodeEncodeError on Windows

The scripts handle UTF-8 encoding automatically. If you still see issues, set:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### Issue: Missing NLTK Data

The script automatically downloads required NLTK data. If it fails:
```python
import nltk
nltk.download('punkt')
```

### Issue: Groq API Errors

Make sure your Groq API key is set in `cookmate_rag.py`:
```python
api_key = os.getenv("YOUR_GROQ_API_KEY")
```

### Issue: Slow Evaluation

- Evaluation of 100 questions may take 10-30 minutes depending on API response times
- Consider testing with fewer questions first: `python generate_qa_dataset.py 10`

## Example Output

```
======================================================================
  CookMate RAG Evaluation System
======================================================================

Step 1: Generating QA Dataset
Generating 100 Question-Answer pairs from recipes.json...
Generated 100 QA pairs
Saved to: rag_qa_dataset.json

Step 2: Initializing Models
🤖 Initializing RAG Model...
✅ RAG Model ready!

Step 3: Evaluating Models
...

📈 Performance Comparison Summary:

Metric                          RAG            Baseline        Improvement    
---------------------------------------------------------------------------
exact_match_rate                0.4500          0.1200          +275.00%
avg_semantic_similarity         0.7823          0.4521          +73.10%
avg_rougeL                      0.6234          0.3456          +80.38%

✅ Evaluation Complete!
```

## Citation

If you use this evaluation system, please cite:
- Dataset: CookMate RAG Evaluation Dataset (recipes.json)
- Models: Groq LLM (llama-3.1-8b-instant)
- Metrics: BLEU, ROUGE, Semantic Similarity

