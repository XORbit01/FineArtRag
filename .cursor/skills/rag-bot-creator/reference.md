# RAG Bot Creator - Advanced Reference

## Advanced Retrieval Techniques

### Query Expansion
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Parent Document Retriever
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
```

### Self-Query Retriever
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source document",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="The publication date",
        type="date",
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Description of documents",
    metadata_field_info=metadata_field_info,
)
```

## Advanced Chunking Strategies

### Semantic Chunking
```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)
chunks = semantic_splitter.create_documents([text])
```

### Code-Aware Chunking
```python
from langchain.text_splitter import Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)
```

## Multi-Modal RAG

### Image + Text
```python
from langchain.document_loaders import UnstructuredImageLoader
from langchain.vectorstores import Qdrant

# Load images with OCR
image_loader = UnstructuredImageLoader("image.png")
image_docs = image_loader.load()

# Combine with text documents
all_docs = text_docs + image_docs
```

## Advanced Prompting

### Chain-of-Thought RAG
```python
prompt_template = """Answer the question step by step.

Step 1: Identify key information from the context
Step 2: Analyze the relationships
Step 3: Synthesize the answer

Context: {context}
Question: {question}

Answer:"""
```

### Reflective RAG
```python
# First pass: Generate answer
initial_answer = qa_chain({"query": question})

# Second pass: Verify and refine
verification_prompt = f"""
Original question: {question}
Initial answer: {initial_answer['result']}
Context used: {initial_answer['source_documents']}

Verify: Is this answer accurate and complete? If not, refine it.
"""
```

## Ollama-Specific Optimizations

### Model Selection for Different Tasks
```python
# General purpose RAG
llm = ChatOllama(model="llama3", base_url="http://localhost:11434")

# Code-focused RAG
llm = ChatOllama(model="codellama", base_url="http://localhost:11434")

# Fast, lightweight RAG
llm = ChatOllama(model="phi3", base_url="http://localhost:11434")

# Multilingual RAG
llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
```

### Custom Context Window
```python
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    num_ctx=8192,  # Increase for longer context
    num_predict=512  # Max tokens to generate
)
```

### GPU Acceleration
```python
# Ollama automatically uses GPU if available
# Check GPU usage:
# ollama ps

# Force CPU (if needed):
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    num_gpu=0  # Use CPU only
)
```

### Model Quantization
```python
# Use quantized models for faster inference
# Pull quantized version:
# ollama pull llama3:8b-q4_0  # 4-bit quantization

llm = ChatOllama(
    model="llama3:8b-q4_0",  # Smaller, faster model
    base_url="http://localhost:11434"
)
```

### Ollama Embeddings Batch Processing
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Batch embeddings for better performance
texts = [chunk.page_content for chunk in chunks]
batch_embeddings = embeddings.embed_documents(texts)
```

### Multiple Ollama Instances
```python
# Run different models for different tasks
qa_llm = ChatOllama(model="llama3", base_url="http://localhost:11434")
summarizer_llm = ChatOllama(model="mistral", base_url="http://localhost:11435")
```

### Ollama Streaming with Custom Handler
```python
from langchain.callbacks.base import BaseCallbackHandler

class OllamaStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    streaming=True,
    callbacks=[OllamaStreamHandler()]
)
```

## Performance Optimization

### Async Processing
```python
import asyncio
from langchain.vectorstores import Chroma

async def async_embed_and_store(docs):
    embeddings = await embeddings.aembed_documents([doc.page_content for doc in docs])
    await vectorstore.aadd_texts(texts=[doc.page_content for doc in docs], embeddings=embeddings)

asyncio.run(async_embed_and_store(chunks))
```

### Batch Embedding
```python
# Process in batches
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_embeddings = embeddings.embed_documents(batch)
    vectorstore.add_texts(batch, batch_embeddings)
```

## Security Considerations

### API Key Management
```python
import os
from langchain.utils import get_from_env

# Use environment variables
api_key = get_from_env("OPENAI_API_KEY", "your-key-here")

# Or use secrets management
import boto3
secrets = boto3.client('secretsmanager')
api_key = secrets.get_secret_value(SecretId='openai-key')['SecretString']
```

### Input Sanitization
```python
def sanitize_query(query: str) -> str:
    # Remove potential injection attempts
    query = query.replace("'", "").replace('"', '')
    # Limit length
    return query[:500]
```

## Monitoring and Logging

### Query Logging
```python
import logging
from langchain.callbacks import StdOutCallbackHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryLogger(StdOutCallbackHandler):
    def on_retriever_end(self, documents, **kwargs):
        logger.info(f"Retrieved {len(documents)} documents")
    
    def on_llm_end(self, response, **kwargs):
        logger.info(f"Generated response: {response.generations[0][0].text[:100]}")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[QueryLogger()]
)
```
