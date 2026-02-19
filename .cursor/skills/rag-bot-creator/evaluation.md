# RAG System Evaluation

## Evaluation Metrics

### Retrieval Metrics

**Precision@K**: Fraction of retrieved docs that are relevant
```python
def precision_at_k(retrieved_docs, relevant_docs, k):
    top_k = retrieved_docs[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_docs))
    return relevant_retrieved / k
```

**Recall@K**: Fraction of relevant docs that were retrieved
```python
def recall_at_k(retrieved_docs, relevant_docs, k):
    top_k = retrieved_docs[:k]
    relevant_retrieved = len(set(top_k) & set(relevant_docs))
    return relevant_retrieved / len(relevant_docs)
```

**MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant doc
```python
def mrr(retrieved_docs, relevant_docs):
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0
```

### Generation Metrics

**Faithfulness**: Answer is grounded in retrieved context
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("labeled_criteria", criteria="faithfulness")
result = evaluator.evaluate_strings(
    prediction=answer,
    reference=context,
    input=question
)
```

**Relevance**: Answer is relevant to the question
```python
evaluator = load_evaluator("labeled_criteria", criteria="relevance")
result = evaluator.evaluate_strings(
    prediction=answer,
    reference=question,
    input=question
)
```

**Answer Correctness**: Answer matches ground truth
```python
evaluator = load_evaluator("labeled_criteria", criteria="correctness")
result = evaluator.evaluate_strings(
    prediction=answer,
    reference=ground_truth,
    input=question
)
```

## Evaluation Dataset

### Create Test Set
```python
test_questions = [
    {
        "question": "What is the main topic?",
        "expected_answer": "The main topic is...",
        "relevant_docs": ["doc1.pdf", "doc2.pdf"]
    },
    # More test cases...
]
```

### Run Evaluation
```python
def evaluate_rag_system(qa_chain, test_questions):
    results = []
    
    for test_case in test_questions:
        # Get answer
        result = qa_chain({"query": test_case["question"]})
        answer = result["result"]
        retrieved_docs = [doc.metadata.get("source") for doc in result["source_documents"]]
        
        # Calculate metrics
        precision = precision_at_k(retrieved_docs, test_case["relevant_docs"], k=5)
        recall = recall_at_k(retrieved_docs, test_case["relevant_docs"], k=5)
        
        # Evaluate answer quality
        faithfulness = evaluator.evaluate_strings(
            prediction=answer,
            reference=" ".join([doc.page_content for doc in result["source_documents"]]),
            input=test_case["question"]
        )
        
        results.append({
            "question": test_case["question"],
            "precision": precision,
            "recall": recall,
            "faithfulness": faithfulness["score"]
        })
    
    return results
```

## A/B Testing

### Compare Configurations
```python
configs = {
    "baseline": {
        "chunk_size": 1000,
        "top_k": 5,
        "embedding_model": "text-embedding-ada-002"
    },
    "optimized": {
        "chunk_size": 500,
        "top_k": 10,
        "embedding_model": "text-embedding-3-small"
    }
}

results = {}
for config_name, config in configs.items():
    qa_chain = create_rag_chain(config)
    results[config_name] = evaluate_rag_system(qa_chain, test_questions)
```

## Monitoring Production

### Track Metrics
```python
import time
from datetime import datetime

def log_query(query, answer, retrieved_docs, response_time):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer_length": len(answer),
        "num_docs_retrieved": len(retrieved_docs),
        "response_time_ms": response_time * 1000
    }
    # Send to monitoring system (e.g., Prometheus, Datadog)
    logger.info(log_entry)
```

### User Feedback
```python
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    # Store feedback for analysis
    feedback_db.insert({
        "query": feedback.query,
        "answer": feedback.answer,
        "rating": feedback.rating,  # 1-5 stars
        "helpful": feedback.helpful  # True/False
    })
```
