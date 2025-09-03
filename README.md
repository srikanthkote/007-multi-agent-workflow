# Multi-Agent Travel Planning System

A sophisticated multi-agent system for comprehensive travel planning, leveraging LangChain and LangGraph to coordinate specialized AI agents for hotel bookings, flight arrangements, and destination guidance.

## Features

- **Specialized Travel Agents**: Dedicated agents for hotels, flights, and travel guidance
- **Seamless Integration**: Coordinated workflow between agents for end-to-end travel planning
- **Natural Language Interface**: Intuitive conversation-based interaction
- **Extensible Architecture**: Easy to add new agents or customize existing ones
- **RAG-Enhanced**: Utilizes Retrieval-Augmented Generation for accurate, up-to-date information

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd windsurf-project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### 1. OpenAI API Key
You'll need an OpenAI API key to use the language models:
- Sign up at [OpenAI](https://platform.openai.com/)
- Generate an API key from your account settings
- Set it as an environment variable:
  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```
  Or add it to a `.env` file in the project root:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```

### 2. Optional: Knowledge Base
- The system will automatically create a knowledge base directory at `./data/knowledge_base`
- You can add custom destination information by placing files in this directory

## Usage

### Command-Line Interface
```bash
python cli.py
```

### Available Commands
- Start an interactive session: `python cli.py`
- Enable verbose logging: `python cli.py --verbose`
- Use a custom config: `python cli.py --config /path/to/config.json`

### Example Queries
- "Find me a hotel in Paris for next weekend"
- "What are the top attractions in Tokyo?"
- "Book a flight from New York to London in July"
- "Plan a 3-day trip to Rome including flights and hotels"
- "What's the best time to visit Japan?"

## Architecture

The system is built around a modular architecture with specialized agents:

### Core Components

- **`BaseAgent`**: Abstract base class for all agents
- **`HotelAgent`**: Handles hotel searches and bookings
- **`FlightAgent`**: Manages flight searches and reservations
- **`GuideAgent`**: Provides destination information and recommendations
- **`TravelOrchestrator`**: Coordinates between different agents
- **`CLI`**: Command-line interface for user interaction

### Agent Capabilities

#### Hotel Agent
- Search for hotels based on location, dates, and preferences
- Book hotel rooms
- Filter by amenities, price range, and ratings

#### Flight Agent
- Search for flights between locations
- Check flight status
- Book flights
- Filter by price, airline, and number of stops

#### Guide Agent
- Provide destination overviews
- Suggest attractions and activities
- Create customized itineraries
- Offer local tips and recommendations

## Configuration

The system can be configured using a JSON configuration file. By default, it looks for `config.json` in the project root.

### Example Config
```json
{
  "llm": {
    "model_name": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 2000
  },
  "vector_store": {
    "persist_directory": "./data/vector_store",
    "collection_name": "travel_knowledge"
  },
  "knowledge_base": {
    "path": "./data/knowledge_base"
  }
}
```

## Extending the System

### Adding a New Agent
1. Create a new class that inherits from `BaseAgent`
2. Implement the required methods (`initialize_agent`, `process_query`)
3. Add any agent-specific tools
4. Register the agent in the `TravelOrchestrator`

### Customizing Existing Agents
- Override methods in the existing agent classes
- Add new tools or modify existing ones
- Update the agent's system message for different behavior

## Dependencies

- Python 3.9+
- LangChain
- LangGraph
- OpenAI API access
- ChromaDB (for vector storage)
- Rich (for CLI formatting)
- Python-dotenv (for environment variables)

## Troubleshooting

1. **API Key Issues**:
   - Ensure your OpenAI API key is set correctly
   - Check for typos in the environment variable name

2. **Installation Problems**:
   - Make sure you're using Python 3.9 or higher
   - Try reinstalling the requirements: `pip install -r requirements.txt --force-reinstall`

3. **Performance Issues**:
   - Reduce the number of results or complexity of queries
   - Use a smaller language model (e.g., gpt-3.5-turbo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Uses [OpenAI](https://openai.com/) for language models
- Inspired by modern AI agent architectures

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (default, lightweight)
- `sentence-transformers/all-mpnet-base-v2` (higher quality)
- `sentence-transformers/all-distilroberta-v1` (balanced performance)

### Language Models
- `microsoft/DialoGPT-medium` (default)
- Custom HuggingFace models supported

### Search Parameters
- **Chunk Size**: 256 tokens (default)
- **Chunk Overlap**: 50 tokens (default)
- **Retrieval Count**: 3-4 documents (configurable)
- **Search Type**: MMR (Maximum Marginal Relevance)

## Example Output

The system provides detailed output including:
- Document processing progress
- Similarity search results with scores
- Retrieved document content and metadata
- Generated answers in formatted tables

## Requirements

- Python 3.8+
- HuggingFace API token
- PDF documents to search
- Sufficient memory for embedding models (2GB+ recommended)

## GPU Support

For faster processing with CUDA-enabled GPUs:
1. Uncomment GPU-related packages in `requirements.txt`
2. Install PyTorch with CUDA support
3. The system will automatically detect and use GPU acceleration

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce chunk size or use smaller embedding models
2. **API Token Issues**: Ensure your HuggingFace token is valid and has appropriate permissions
3. **PDF Loading Errors**: Check PDF file integrity and permissions
4. **Model Download Issues**: Ensure stable internet connection for initial model downloads

### Performance Tips

- Use GPU acceleration for large document collections
- Adjust chunk size based on document complexity
- Consider using smaller embedding models for faster processing
- Implement caching for frequently accessed embeddings

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the semantic search pipeline.

## License

This project is open source and available under the MIT License.
