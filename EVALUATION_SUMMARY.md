# RAG Evaluation System - Summary

## What Was Created

A comprehensive RAG evaluation system has been implemented to compare RAG-based models vs baseline models (without RAG) on 100 Question-Answer pairs.

## Files Created

### 1. `generate_qa_dataset.py`
- **Purpose**: Generates 100 Question-Answer pairs from `recipes.json`
- **Features**:
  - Creates diverse question types (recipe overview, ingredients, steps, multi-recipe, timing, cooking techniques)
  - Includes document name (`recipes.json`) and page/section references for each QA pair
  - Handles questions requiring information from multiple recipes
  - Saves dataset in JSON format with metadata

### 2. `baseline_model.py`
- **Purpose**: Implements a baseline model that uses LLM without RAG
- **Features**:
  - Uses Groq LLM without access to recipe knowledge base
  - Generates answers based on general knowledge only
  - Provides fair comparison baseline for RAG model

### 3. `rag_evaluation.py`
- **Purpose**: Comprehensive evaluation system with multiple metrics
- **Metrics Implemented**:
  - **Exact Match**: Binary score for exact answer matches
  - **Partial Match (F1)**: Word overlap F1 score
  - **BLEU Score**: Bilingual Evaluation Understudy score
  - **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L (unigram, bigram, longest common subsequence)
  - **Semantic Similarity**: Cosine similarity between embeddings
  - **Latency**: Response time measurement
- **Features**:
  - Evaluates both RAG and baseline models
  - Compares performance and calculates improvements
  - Generates comprehensive reports
  - Exports results to CSV for analysis

### 4. `run_rag_evaluation.py`
- **Purpose**: Main script to run complete evaluation pipeline
- **Features**:
  - Automatically generates or loads QA dataset
  - Initializes both models
  - Runs evaluation on all 100 questions
  - Compares results and generates reports
  - Provides detailed output and summary

### 5. `RAG_EVALUATION_README.md`
- **Purpose**: Comprehensive documentation
- **Contents**:
  - Usage instructions
  - Dataset structure explanation
  - Metric descriptions
  - Troubleshooting guide
  - Example outputs

## Dataset Structure

The generated dataset (`rag_qa_dataset.json`) contains:

- **100 QA pairs** with:
  - Question text
  - Ground truth answer
  - Document name: `recipes.json`
  - Page/section references (e.g., "Page 1 (Recipe Overview Section)")
  - Recipe ID and name
  - Question type classification

- **Question Types Distribution**:
  - Recipe Overview: 20 questions
  - Ingredients: 25 questions
  - Steps: 30 questions
  - Multi-recipe: 10 questions
  - Timing: 10 questions
  - Cooking Techniques: 5 questions

## How to Use

### Quick Start

1. **Generate Dataset** (if not already generated):
   ```bash
   python generate_qa_dataset.py
   ```

2. **Run Full Evaluation**:
   ```bash
   python run_rag_evaluation.py
   ```

### Output Files

After running evaluation, you'll get:

1. **`rag_evaluation_report.json`**: Complete evaluation report with:
   - Aggregated metrics for both models
   - Per-question detailed results
   - Comparison and improvement percentages
   - Summary findings

2. **`rag_evaluation_results.csv`**: Detailed results in CSV format for:
   - Statistical analysis
   - Visualization
   - Further processing

## Evaluation Metrics Explained

### Exact Match Rate
- Percentage of answers that exactly match ground truth
- **Expected**: RAG should have higher rate (access to specific recipe data)

### Semantic Similarity
- Measures how semantically similar answers are (0-1 scale)
- **Expected**: RAG should score higher (context-aware answers)

### ROUGE-L
- Measures answer quality and completeness
- **Expected**: RAG should show better scores (more accurate information)

### Latency
- Response time per question
- **Note**: RAG may be slightly slower due to retrieval step

## Requirements

All dependencies are in `requirements.txt`. Key additions:
- `nltk==3.8.1` - For BLEU score calculation
- `rouge-score==0.1.2` - For ROUGE metrics

## Key Features

✅ **100 QA Pairs**: Comprehensive dataset covering all recipe aspects
✅ **Document References**: Each QA pair includes source document and page references
✅ **Multi-Document Support**: Handles questions requiring information from multiple recipes
✅ **Multiple Metrics**: 7 different evaluation metrics for comprehensive comparison
✅ **Automated Pipeline**: Single command to run complete evaluation
✅ **Detailed Reports**: JSON and CSV outputs for analysis
✅ **Windows Compatible**: Handles UTF-8 encoding issues

## Next Steps

1. Run the evaluation: `python run_rag_evaluation.py`
2. Review the report: `rag_evaluation_report.json`
3. Analyze results: `rag_evaluation_results.csv`
4. Compare RAG vs Baseline performance
5. Document findings and improvements

## Notes

- Evaluation of 100 questions may take 10-30 minutes depending on API response times
- Make sure your Groq API key is configured in `cookmate_rag.py`
- For faster testing, generate fewer questions: `python generate_qa_dataset.py 10`

