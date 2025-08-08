# Conversational Search with RAG

A conversational search system implementing Retrieval-Augmented Generation (RAG) for the Information Retrieval course. This project combines classical IR methods with modern neural approaches to enable context-aware document retrieval and query rewriting.

## Overview

This system implements a multi-stage retrieval pipeline that processes conversational queries by:
- Building conversation context across multiple turns
- Retrieving relevant documents using BM25 and Language Models
- Re-ranking results with neural cross-encoders
- Generating responses using LLM integration

## Key Features

- **Multi-stage Retrieval**: BM25 → Language Model with Dirichlet Smoothing → Cross-encoder re-ranking
- **Conversational Context**: Accumulates query history for context-aware search
- **Neural Query Rewriting**: LLM-powered query reformulation using Llama 3.2
- **BERT Analysis**: Attention visualization and embedding-based relevance classification
- **TREC CAsT Evaluation**: Standard conversational search metrics (P@10, NDCG@5, AP)

## Project Structure

```
├── Project Conversational Search.ipynb    # Main RAG pipeline implementation
├── Llama-API.ipynb                        # LLM integration for query rewriting
├── OpenSearchSimpleAPI.py                 # Search engine interface
├── TRECCASTeval.py                        # Evaluation framework
├── rank_metric.py                         # IR metrics implementation
├── data/                                  # TREC CAsT dataset
│   ├── training/                          # Training topics and relevance judgments
│   └── evaluation/                        # Test topics and relevance judgments
└── Results/                               # Generated performance plots
```

## Technical Implementation

### Retrieval Pipeline
```python
# Stage 1: Initial retrieval with BM25
opensearch_results = opensearch.search_body(query, numDocs=k)

# Stage 2: Language Model scoring with Dirichlet smoothing
LMD_score = (term_freq + mu * collection_prob) / (doc_length + mu)

# Stage 3: Neural re-ranking with cross-encoders
model = AutoModelForSequenceClassification.from_pretrained(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)
```

### Query Processing
- **Preprocessing**: Tokenization, stemming, stopword removal
- **Context Accumulation**: Concatenates previous utterances for conversation history
- **Neural Rewriting**: Uses Llama 3.2 for query reformulation

### Evaluation
Evaluated on TREC CAsT dataset using standard IR metrics:
- **Precision@10**: 0.8 (BM25) vs 0.4 (LMD) for topic 77, turn 1
- **NDCG@5**: 0.622 (BM25) vs 0.051 (LMD) for topic 77, turn 1
- **Query Rewriting**: 0.73 TF-IDF similarity, 0.85 BERT similarity with originals

## Models Used

- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **BERT**: `bert-base-uncased` for embeddings and classification
- **LLM**: Llama 3.2 via Ollama API for query rewriting

## Setup

1. Install dependencies:
```bash
pip install opensearch-py transformers torch scikit-learn pandas matplotlib seaborn nltk spacy
```

2. Configure OpenSearch credentials in `OpenSearchSimpleAPI.py`

3. Run the main notebook: `Project Conversational Search.ipynb`

## Results

The system demonstrates that:
- BM25 consistently outperforms Language Models across most metrics
- Cross-encoder re-ranking provides measurable improvements
- Neural query rewriting maintains semantic similarity while improving retrieval
- Performance varies significantly across conversation turns

Generated visualizations in `Results/` show comparative performance across different retrieval methods.

---

*Academic project for Information Retrieval course (2022-2023)*