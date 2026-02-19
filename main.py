#!/usr/bin/env python3
"""
Main entry point for Fine Arts RAG System
"""

import sys
import argparse
from pathlib import Path

from src.config.settings import AppConfig, config
from src.utils.logger import setup_logger
from src.utils.ollama_check import OllamaHealthCheck
from src.rag.system import RAGSystem
from src.cli.interface import CLIInterface


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Fine Arts RAG System - Lebanese University",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--pages-dir",
        type=str,
        default=None,
        help="Directory containing text documents"
    )
    parser.add_argument(
        "--chroma-db",
        type=str,
        default=None,
        help="ChromaDB persistence directory"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Ollama embedding model"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Ollama LLM model"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Force recreate vector store"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to execute"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run FastAPI server instead of CLI"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API host (used with --api)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (used with --api)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Override config with command-line arguments
    if args.pages_dir:
        config.documents.pages_directory = Path(args.pages_dir)
    if args.chroma_db:
        config.vector_store.persist_directory = Path(args.chroma_db)
    if args.embedding_model:
        config.embeddings.model_name = args.embedding_model
    if args.llm_model:
        config.ollama.llm_model = args.llm_model
    
    config.logging.level = args.log_level
    if args.log_file:
        config.logging.file = Path(args.log_file)
    
    # Setup logging
    logger = setup_logger(__name__, config.logging)
    logger.info("Starting Fine Arts RAG System")

    if args.api:
        from src.api.app import create_app
        import uvicorn

        app = create_app(config, force_recreate=args.recreate)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower()
        )
        return
    
    # Check Ollama connection
    health_check = OllamaHealthCheck(config.ollama)
    if not health_check.check_connection():
        logger.warning("Ollama service not accessible. Some features may not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            logger.info("Exiting due to Ollama unavailability")
            sys.exit(1)
    
    # Initialize RAG system
    try:
        rag_system = RAGSystem(config)
        rag_system.initialize(force_recreate=args.recreate)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Create CLI interface
    cli = CLIInterface(rag_system)
    
    # Execute query or start interactive session
    if args.query:
        cli.run_single_query(args.query)
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()
