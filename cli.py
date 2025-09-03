#!/usr/bin/env python3
"""
Command-line interface for the Multi-Agent Travel Planning System.
"""
import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler

from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from orchestrator import TravelOrchestrator

# Set up console for rich output
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("travel_planner")

class TravelPlannerCLI:
    """Command-line interface for the travel planning system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the CLI.
        
        Args:
            config_path: Path to a JSON configuration file.
        """
        self.config = self._load_config(config_path)
        self.llm = self._initialize_llm()
        self.vector_store = self._initialize_vector_store()
        self.orchestrator = None
        self.chat_history = []
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "llm": {
                "model_name": "meta-llama/Llama-3.1-8B",
                "use_hf_hub": True,  # Set to False to use local models
                "temperature": 0.7,
                "max_tokens": 500
            },
            "vector_store": {
                "persist_directory": "./data/vector_store",
                "collection_name": "travel_knowledge"
            },
            "knowledge_base": {
                "path": "./data/knowledge_base"
            },
            "ui": {
                "welcome_message": "Welcome to the AI Travel Planner! How can I assist with your travel plans today?"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    # Merge with defaults
                    import json
                    user_config = json.load(f)
                    return self._deep_update(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}. Using defaults.")
        
        return default_config
    
    def _deep_update(self, original: Dict, update: Dict) -> Dict:
        """Recursively update a dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                original[key] = self._deep_update(original[key], value)
            else:
                original[key] = value
        return original
    
    def _initialize_llm(self) -> Any:
        """Initialize the language model."""
        try:
            model_name = self.config["llm"].get("model_name", "google/flan-t5-large")
            
            # Check if using HuggingFace Hub (online)
            if self.config["llm"].get("use_hf_hub", True):
                # Get HuggingFace API token
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not hf_token:
                    logger.warning(
                        "HUGGINGFACEHUB_API_TOKEN environment variable not set. "
                        "Please set it to use HuggingFace Hub models."
                    )
                    return None
                
                return HuggingFaceEndpoint(
                    repo_id=model_name,
                    task="text-generation",
                    temperature=self.config["llm"].get("temperature", 0.7),
                    max_new_tokens=self.config["llm"].get("max_tokens", 500),
                    return_full_text= False,
                    huggingfacehub_api_token=hf_token,
                    streaming=False  # Disable streaming for now to avoid async issues
                )
            else:
                # Local model loading
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                from langchain.llms import HuggingFacePipeline
                
                logger.info(f"Loading model {model_name} locally...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=self.config["llm"].get("max_tokens", 500),
                    temperature=self.config["llm"].get("temperature", 0.7),
                )
                
                return HuggingFacePipeline(pipeline=pipe)
                
        except ImportError as e:
            logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            return None
    
    def _initialize_vector_store(self) -> Any:
        """Initialize the vector store for RAG."""
        try:
            # Create directory if it doesn't exist
            persist_dir = self.config["vector_store"]["persist_directory"]
            os.makedirs(persist_dir, exist_ok=True)
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings()
            
            # Initialize Chroma
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=self.config["vector_store"]["collection_name"]
            )
        except Exception as e:
            logger.warning(f"Failed to initialize vector store: {str(e)}. Some features may be limited.")
            return None
    
    async def initialize(self) -> bool:
        """Initialize the travel planner components with timeout and error handling."""
        if not self.llm:
            logger.error("Language model not initialized. Check configuration and logs.")
            return False
        
        try:
            # Create knowledge base directory
            kb_path = self.config["knowledge_base"]["path"]
            try:
                os.makedirs(kb_path, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create knowledge base directory: {str(e)}")
                return False
            
            # Initialize orchestrator with timeout
            try:
                self.orchestrator = await asyncio.wait_for(
                    asyncio.to_thread(
                        TravelOrchestrator,
                        llm=self.llm,
                        vector_store=self.vector_store,
                        knowledge_base_path=kb_path
                    ),
                    timeout=30.0  # 30 second timeout for initialization
                )
                logger.info("Travel planner initialized successfully!")
                return True
                
            except asyncio.TimeoutError:
                logger.error("Initialization timed out after 30 seconds")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize travel planner: {str(e)}", exc_info=True)
            return False
    
    async def process_query(self, query: str) -> str:
        """Process a user query through the orchestrator with timeout and error handling."""
        if not self.orchestrator:
            logger.error("Orchestrator not initialized")
            return "Error: Service not ready. Please try again."
        
        try:
            #with console.status("Thinking...", spinner="dots"):
            try:
                response = await self.orchestrator.process_query(
                        query=query,
                        context={"chat_history": self.chat_history}
                    )
    
                if response and hasattr(response, 'success'):
                    if response.success:
                        #print("cli.py:successful response:" + str(response))
                        self.chat_history.append(HumanMessage(content=query))
                        output = response.data.get("output", "No response generated")
                        self.chat_history.append(AIMessage(content=output))
                        return output
                    return f"Error: {response.error or 'Unknown error'}"
                return "Error: Invalid response format"
                
            except asyncio.TimeoutError:
                logger.error("Query processing timed out")
                return "Error: Request timed out. Please try again."
            except Exception as e:
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                raise
                    
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return "Error: An unexpected error occurred. Please try again later."
    
    async def interactive_session(self):
        """Start an interactive session with the travel planner."""
        # Print welcome message
        console.print(
            Panel.fit(
                Markdown(self.config["ui"]["welcome_message"]),
                title="🤖 AI Travel Planner",
                border_style="blue"
            )
        )
        
        # Main interaction loop
        try:
            # Get user input
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
            
            # Check for exit commands
            if user_input.lower() in ('exit', 'quit', 'bye'):
                console.print("\n[bold green]Goodbye! Safe travels! ✈️[/bold green]")
            
            # Process the query
            response = await self.process_query(user_input)
            # Display the response
            console.print("\n[bold green]Assistant:[/bold green]")
            #console.print(Markdown(response))
            console.print(response)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Use 'exit' or 'quit' to end the session.[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            logger.exception("An unexpected error occurred")

@click.command()
@click.option('--config', '-c', 'config_path', 
              help='Path to a JSON configuration file.',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
def main(config_path: Optional[str], verbose: bool):
    """Run the Travel Planner CLI."""
    # Set log level
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create and run the CLI
    async def run():
        cli = TravelPlannerCLI(config_path)
        if await cli.initialize():
            await cli.interactive_session()
    
    try:
        asyncio.run(run())
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        logger.exception("Application crashed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
