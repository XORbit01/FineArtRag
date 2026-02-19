# RAG Bot Deployment Patterns

## Web Application (Flask/FastAPI)

### FastAPI Example
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA

app = FastAPI()

# Initialize RAG components (do once at startup)
qa_chain = None

@app.on_event("startup")
async def startup_event():
    global qa_chain
    qa_chain = initialize_rag_system()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = qa_chain({"query": request.query})
    return QueryResponse(
        answer=result["result"],
        sources=[doc.metadata.get("source", "") for doc in result["source_documents"]]
    )
```

## Streamlit Application

```python
import streamlit as st
from langchain.chains import RetrievalQA

st.title("RAG Chatbot")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = initialize_rag_system()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    result = st.session_state.qa_chain({"query": prompt})
    
    with st.chat_message("assistant"):
        st.markdown(result["result"])
        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")
    
    st.session_state.messages.append({"role": "assistant", "content": result["result"]})
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./data:/app/data
```

## Cloud Deployment

### AWS Lambda
```python
import json
from langchain.chains import RetrievalQA

def lambda_handler(event, context):
    query = event.get('query', '')
    
    # Initialize (consider caching in Lambda container)
    qa_chain = initialize_rag_system()
    
    result = qa_chain({"query": query})
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'answer': result['result'],
            'sources': [doc.metadata.get('source') for doc in result['source_documents']]
        })
    }
```

### Google Cloud Functions
```python
from langchain.chains import RetrievalQA

qa_chain = None

def initialize():
    global qa_chain
    if qa_chain is None:
        qa_chain = initialize_rag_system()

def rag_query(request):
    initialize()
    query = request.json.get('query', '')
    result = qa_chain({"query": query})
    return {'answer': result['result']}
```

## Production Considerations

### Caching
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# Or use Redis
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis()
set_llm_cache(RedisCache(redis_client))
```

### Rate Limiting
```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query: QueryRequest):
    # RAG logic here
    pass
```

### Health Checks
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vectorstore": "connected" if vectorstore else "disconnected",
        "llm": "connected" if llm else "disconnected"
    }
```

### Error Handling
```python
from fastapi import HTTPException

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = qa_chain({"query": request.query})
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
```
