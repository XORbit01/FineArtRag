---
name: rag-bot-creator
description: Builds RAG (Retrieval-Augmented Generation) bots and chatbots using Ollama offline models, vector databases, embeddings, and LLM integration. Use when creating RAG systems, building knowledge bases, implementing semantic search, working with vector databases, or integrating retrieval with language models. Prioritizes offline/local models via Ollama.
---

# RAG Bot Creator Engineer

## Quick Start

RAG systems combine retrieval (finding relevant context) with generation (LLM responses). Core components:

1. **Document Processing**: Load and chunk documents
2. **Embeddings**: Convert text to vector representations
3. **Vector Store**: Store and search embeddings
4. **Retrieval**: Find relevant context for queries
5. **Generation**: Use LLM to generate responses with context

## Architecture Overview

```
User Query → Embed Query → Vector Search → Retrieve Context → LLM Generation → Response
```

## Complete Ollama RAG Example

**Full offline RAG bot setup:**
```python
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 2. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings (Ollama)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 6. Create LLM (Ollama)
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.7,
    num_ctx=4096
)

# 7. Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Query
response = qa_chain({"query": "What is the main topic?"})
print(response["result"])
print("\nSources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata.get('source', 'Unknown')}")
```

## Core Workflow

### 1. Document Ingestion

**Load documents:**
```python
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# Single file
loader = TextLoader("document.txt")
documents = loader.load()

# Directory
loader = DirectoryLoader("./docs", glob="**/*.txt")
documents = loader.load()
```

**Chunk documents:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)
```

### 2. Create Embeddings

**Using Ollama (Recommended for offline):**
```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # or "llama2", "mistral", etc.
    base_url="http://localhost:11434"  # Default Ollama URL
)
```

**Using HuggingFace (Alternative offline):**
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Using OpenAI (Cloud):**
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
```

### 3. Vector Store Setup

**ChromaDB (local):**
```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

**Pinecone (cloud):**
```python
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)
```

**FAISS (in-memory):**
```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./faiss_index")
```

### 4. Retrieval Setup

**Basic retrieval:**
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

**MMR (Maximum Marginal Relevance):**
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

**With metadata filtering:**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "document.pdf"}
    }
)
```

### 5. LLM Integration

**Ollama (Recommended for offline RAG bots):**
```python
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

# Chat model (recommended)
llm = ChatOllama(
    model="llama3",  # or "mistral", "codellama", "phi3", etc.
    base_url="http://localhost:11434",
    temperature=0.7,
    num_ctx=4096  # Context window size
)

# Or use LLM interface
llm = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)
```

**Check available Ollama models:**
```bash
ollama list
```

**Pull a model if not installed:**
```bash
ollama pull llama3
ollama pull mistral
ollama pull nomic-embed-text  # For embeddings
```

**OpenAI (Cloud alternative):**
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
```

**Anthropic Claude (Cloud alternative):**
```python
from langchain.chat_models import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

### 6. RAG Chain

**Basic QA Chain:**
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain({"query": "What is the main topic?"})
```

**Conversational RAG:**
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

response = conversational_chain({"question": "Tell me about X"})
```

**Custom prompt:**
```python
from langchain.prompts import PromptTemplate

prompt_template = """Use the following context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)
```

## Best Practices

### Chunking Strategy
- **Chunk size**: 500-1500 tokens (balance context vs precision)
- **Overlap**: 10-20% of chunk size (preserve context across boundaries)
- **For code**: Use code-aware splitters (LangChain's `CodeTextSplitter`)
- **For structured data**: Preserve document structure in metadata

### Embedding Selection
- **Ollama (Recommended)**: Offline, free, runs locally. Use `nomic-embed-text` for embeddings
- **HuggingFace**: Free, good quality, runs locally, no Ollama needed
- **OpenAI**: Best quality, requires API key, costs per token
- **Cohere**: Good multilingual support
- **For offline RAG**: Prefer Ollama embeddings with Ollama LLMs for consistency

### Retrieval Optimization
- **Top-k**: Start with 3-5, increase if context insufficient
- **MMR**: Use when you need diverse results (avoid redundancy)
- **Metadata filtering**: Add filters for source, date, category
- **Re-ranking**: Use cross-encoders for better relevance (e.g., `sentence-transformers`)

### Prompt Engineering
- **Include context clearly**: "Based on the following context..."
- **Specify format**: "Answer in bullet points" or "Provide a summary"
- **Handle uncertainty**: "If the context doesn't contain the answer, say so"
- **Add examples**: Few-shot prompts improve consistency

### Performance Considerations
- **Batch embeddings**: Process multiple documents at once
- **Persist vector stores**: Save embeddings to avoid recomputation
- **Async operations**: Use async APIs for better throughput
- **Caching**: Cache frequent queries and embeddings

## Common Patterns

### Multi-Document RAG
```python
# Load multiple sources
loaders = [
    PyPDFLoader("doc1.pdf"),
    TextLoader("doc2.txt"),
    DirectoryLoader("./docs", glob="*.md")
]

all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())

# Process together
chunks = text_splitter.split_documents(all_docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
```

### Hybrid Search (Keyword + Semantic)
```python
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Keyword search
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Semantic search
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

### Streaming Responses

**With Ollama (Recommended):**
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

**With OpenAI:**
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)
```

### Adding Metadata
```python
# Add metadata during chunking
for i, chunk in enumerate(chunks):
    chunk.metadata = {
        "source": "document.pdf",
        "page": i,
        "section": "introduction"
    }

# Filter by metadata
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"section": "introduction"}
    }
)
```

## Troubleshooting

### Low Quality Retrievals
- **Increase chunk overlap**: More context preserved
- **Adjust chunk size**: Smaller chunks = more precise, larger = more context
- **Try different embedding models**: Some models work better for specific domains
- **Add re-ranking**: Use cross-encoder models to re-rank results

### Hallucinations
- **Improve prompt**: Explicitly tell LLM to only use provided context
- **Add source citations**: Include document sources in response
- **Lower temperature**: Reduce randomness (0.0-0.3 for factual)
- **Validate retrieved context**: Check if retrieved chunks are actually relevant

### Slow Performance
- **Use Ollama offline models**: No API latency, runs on local hardware
- **Use local embeddings**: Ollama or HuggingFace models run faster than API calls
- **Reduce top-k**: Retrieve fewer documents
- **Batch processing**: Process multiple queries together
- **Cache embeddings**: Store computed embeddings
- **Optimize Ollama**: Use smaller models (e.g., `llama3:8b` instead of `llama3:70b`) or quantized versions

## Required Dependencies

```bash
# Core LangChain
pip install langchain langchain-community

# Ollama (for offline models)
pip install ollama  # Python client (optional, LangChain handles it)
# Install Ollama server: https://ollama.ai/download

# Vector stores
pip install chromadb  # Local (recommended for offline)
pip install faiss-cpu  # In-memory alternative
pip install pinecone-client  # Cloud alternative

# Document loaders
pip install pypdf python-docx beautifulsoup4

# Embeddings
pip install sentence-transformers  # HuggingFace (alternative offline)
# For Ollama embeddings, use langchain-community (already installed above)

# Optional cloud providers
pip install openai  # OpenAI API
pip install anthropic  # Anthropic API
```

## Ollama Setup

**1. Install Ollama:**
- Download from https://ollama.ai/download
- Or use: `curl https://ollama.ai/install.sh | sh` (Linux/Mac)

**2. Pull required models:**
```bash
# LLM models
ollama pull llama3          # General purpose (recommended)
ollama pull mistral         # Alternative
ollama pull codellama       # Code-focused
ollama pull phi3            # Small, fast

# Embedding models
ollama pull nomic-embed-text  # For embeddings
```

**3. Verify installation:**
```bash
ollama list
ollama run llama3 "Hello, world!"
```

**4. Start Ollama service:**
Ollama runs automatically on `http://localhost:11434`. Ensure it's running:
```bash
# Check if running
curl http://localhost:11434/api/tags

# Or start manually (if needed)
ollama serve
```

## Additional Resources

- For advanced retrieval techniques, see [reference.md](reference.md)
- For deployment patterns, see [deployment.md](deployment.md)
- For evaluation metrics, see [evaluation.md](evaluation.md)
