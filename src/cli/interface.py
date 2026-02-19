"""
Command-line interface for RAG system
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from src.rag.system import RAGSystem
from src.config.settings import AppConfig

logger = logging.getLogger(__name__)
console = Console()


class CLIInterface:
    """Command-line interface for interacting with RAG system"""
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize CLI interface
        
        Args:
            rag_system: Initialized RAG system instance
        """
        self.rag_system = rag_system
        self.show_sources = True
        self.use_memory = True
    
    def display_answer(self, answer: str, sources: Optional[list] = None) -> None:
        """
        Display answer and sources
        
        Args:
            answer: Generated answer
            sources: List of source documents
        """
        console.print("\n[bold cyan]Answer:[/bold cyan]")
        console.print(Markdown(answer))
        
        if self.show_sources and sources:
            console.print(f"\n[bold cyan]Sources ({len(sources)}):[/bold cyan]")
            for i, doc in enumerate(sources, 1):
                source_file = doc.metadata.get("source_file", "unknown")
                source_url = doc.metadata.get("url", "")
                preview = doc.page_content[:100].replace("\n", " ")
                citation = source_url if source_url else source_file
                console.print(f"  {i}. {citation}")
                console.print(f"     [dim]{preview}...[/dim]")
    
    def run_interactive(self) -> None:
        """Run interactive chat session"""
        console.print(Panel.fit(
            "[bold blue]Fine Arts RAG System[/bold blue]\n"
            "Ask questions about the Fine Arts Faculty.\n"
            "Commands: 'quit'/'exit', 'sources on/off', 'memory on/off', "
            "'memory clear', 'memory status'"
        ))
        
        while True:
            try:
                question = Prompt.ask("\n[bold green]Question[/bold green]")
                
                if question.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Exiting...[/yellow]")
                    break
                
                if question.lower().startswith("sources"):
                    self._handle_sources_command(question)
                    continue

                if question.lower().startswith("memory"):
                    self._handle_memory_command(question)
                    continue
                
                if not question.strip():
                    continue
                
                result = self.rag_system.query(
                    question,
                    return_sources=True,
                    use_memory=self.use_memory
                )
                self.display_answer(
                    result.get("answer", ""),
                    result.get("sources", [])
                )
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                console.print(f"[red]Error: {e}[/red]")
    
    def _handle_sources_command(self, command: str) -> None:
        """Handle sources toggle command"""
        parts = command.split()
        if len(parts) > 1:
            if parts[1].lower() == "off":
                self.show_sources = False
                console.print("[yellow]Source display disabled[/yellow]")
            elif parts[1].lower() == "on":
                self.show_sources = True
                console.print("[yellow]Source display enabled[/yellow]")

    def _handle_memory_command(self, command: str) -> None:
        """Handle memory controls for interactive chat."""
        parts = command.split()
        if len(parts) == 1:
            status = self.rag_system.get_memory_status()
            console.print(
                f"[yellow]Memory enabled={status['enabled']}, "
                f"turns={status['turns']}, summary={status['summary_present']}[/yellow]"
            )
            return

        action = parts[1].lower()
        if action == "off":
            self.use_memory = False
            self.rag_system.set_memory_enabled(False)
            console.print("[yellow]Conversation memory disabled[/yellow]")
        elif action == "on":
            self.use_memory = True
            self.rag_system.set_memory_enabled(True)
            console.print("[yellow]Conversation memory enabled[/yellow]")
        elif action == "clear":
            self.rag_system.clear_memory()
            console.print("[yellow]Conversation memory cleared[/yellow]")
        elif action == "status":
            status = self.rag_system.get_memory_status()
            console.print(
                f"[yellow]Memory enabled={status['enabled']}, "
                f"turns={status['turns']}, summary={status['summary_present']}, "
                f"buffer_chars={status['buffer_chars']}[/yellow]"
            )
    
    def run_single_query(self, question: str) -> None:
        """
        Execute single query and display result
        
        Args:
            question: User question
        """
        try:
            result = self.rag_system.query(
                question,
                return_sources=True,
                use_memory=self.use_memory
            )
            self.display_answer(
                result.get("answer", ""),
                result.get("sources", [])
            )
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
