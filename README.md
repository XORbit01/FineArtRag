# Fine Arts Chat Web App

Lightweight frontend for the RAG API with:
- WebSocket live chat (`/ws/chat`)
- HTTP fallback query (`/v1/query`)
- Fine-arts styled interface (responsive desktop/mobile)

## Chatbot Architecture (Composed Components)

```mermaid
flowchart TB
    %% Presentation
    subgraph UI["Frontend (webapp/)"]
      U["User"]
      FE["Chat UI\n- tabs/history\n- live typing\n- clickable citations"]
      U --> FE
    end

    %% API layer
    subgraph API["API Layer (src/api/)"]
      APP["FastAPI app\n(app.py)"]
      WS["WebSocket Router\n/ws/chat"]
      CHAT["Chats Router\n/v1/chats/*"]
      QRY["Query Router\n/v1/query"]
      SCH["Schemas\n(request/response models)"]
      APP --> WS
      APP --> CHAT
      APP --> QRY
      APP --> SCH
    end

    %% Service layer
    subgraph SVC["Service Layer (RAGApiService)"]
      SRV["RAGApiService\n- session orchestration\n- greeting handling\n- source serialization"]
      MEM["Chat Records (in-memory)\n- title\n- turns\n- rolling summary"]
      WS --> SRV
      CHAT --> SRV
      QRY --> SRV
      SRV <--> MEM
    end

    %% Core RAG
    subgraph CORE["Core RAG (src/rag/)"]
      SYS["RAGSystem\n(query orchestration)"]
      ROU["QueryRouter\n(normalize + intents + metadata filter)"]
      HYB["HybridRetriever\n(Dense + BM25 fusion)"]
      CHN["RAGChainBuilder\n(prompt + chain execution)"]
      SYS --> ROU
      SYS --> HYB
      SYS --> CHN
    end

    %% Data and models
    subgraph DATA["Data + Models"]
      DOCS["pages/*.txt\n(source docs)"]
      VS["Chroma Vector Store"]
      LLM["Ollama LLM"]
      EMB["Embeddings model"]
      DOCS --> VS
      EMB --> VS
      HYB <--> VS
      CHN <--> LLM
    end

    FE <--> APP
    SRV --> SYS
    SYS --> FE
```

## Run

1. Start API:
```bash
python main.py --api --host 127.0.0.1 --port 8000
```

2. Serve this folder:
```bash
python -m http.server 5500 -d webapp
```

3. Open:
```text
http://127.0.0.1:5500
```

The UI auto-connects to `http://<current-host>:8000` and uses backend chat tabs/history.
