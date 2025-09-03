# Multi-Agent Travel Planning System (Demonstration)

A sophisticated multi-agent system for comprehensive travel planning, leveraging LangChain and LangGraph to coordinate specialized AI agents for hotel bookings, flight arrangements, and destination guidance.

> **⚠️ CURRENT STATUS: MOCK IMPLEMENTATION**
>
> Please be aware that this project is currently a **demonstration** of a multi-agent architecture. The core travel booking functionalities (flights and hotels) are **mocked**. The system **does not** connect to real-world APIs like Amadeus to search for or book flights and hotels. The code for a real Amadeus API client exists in the repository but is not currently integrated with the agents. See the "Future Development" section for details on how to connect it.

## Architecture

The system uses a central orchestrator built with LangGraph to manage the conversation and route user queries to the appropriate specialized agent.

```
+-----------------+      +-----------------------+
|                 |      |                       |
|   Interactive   |----->|  TravelOrchestrator   |
|      CLI        |      |      (LangGraph)      |
|                 |      |                       |
+-----------------+      +-----------+-----------+
                         |           |           |
                         |           |           |
           +-------------+           |           +-------------+
           |                         |                         |
           v                         v                         v
+-----------------+      +-----------------+      +-----------------+
|                 |      |                 |      |                 |
|   FlightAgent   |      |   HotelAgent    |      |    GuideAgent   |
|    (Mocked)     |      |    (Mocked)     |      | (Partially RAG) |
|                 |      |                 |      |                 |
+-----------------+      +-----------------+      +-----------------+
```

## Features

-   **Multi-Agent Architecture**: Specialized agents for Flights, Hotels, and Travel Guidance, managed by a central orchestrator.
-   **Keyword-Based Routing**: A simple but effective router in the orchestrator directs user queries to the correct agent.
-   **Interactive CLI**: A user-friendly command-line interface built with Rich and Click for easy interaction.
-   **RAG-Powered Guide Agent**: The `GuideAgent` uses a Retrieval-Augmented Generation (RAG) pipeline with a Chroma vector store and HuggingFace embeddings to answer questions about destinations. (Note: The current knowledge base is very small and hardcoded).
-   **Extensible Design**: The agent-based architecture makes it easy to add new capabilities or connect existing agents to real-world tools.

## Setup and Installation

### 1. Prerequisites

-   Python 3.9+
-   Git

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

The application uses environment variables for API keys. Create a file named `.env` in the root of the project directory and add the following:

```
# For the default HuggingFace LLM
HUGGINGFACEHUB_API_TOKEN="your_hf_api_token_here"

# For connecting to the Amadeus API (see "Future Development")
AMADEUS_API_KEY="your_amadeus_api_key_here"
AMADEUS_API_SECRET="your_amadeus_api_secret_here"

# Optional: For using OpenAI models instead of HuggingFace
# OPENAI_API_KEY="your_openai_api_key_here"
```

-   You **must** provide a `HUGGINGFACEHUB_API_TOKEN` for the application to run with its default configuration. You can get one from the [Hugging Face Hub](https://huggingface.co/settings/tokens).
-   The Amadeus keys are not used by default but are required if you wish to implement the real booking functionality.

## How to Run

Once you have completed the setup, run the interactive CLI from the project's root directory:

```bash
python cli.py
```

You can then start asking travel-related questions like:
-   "Find me a flight from New York to London"
-   "Find me a hotel in Paris"
-   "Tell me about Tokyo"

## How It Works

The application is built around a `TravelOrchestrator` that uses a state machine created with `langgraph`. When you enter a query:

1.  The `cli.py` script captures your input.
2.  It passes the query to the `TravelOrchestrator`.
3.  The orchestrator's routing function inspects the query for keywords (e.g., "flight", "hotel", "attraction").
4.  Based on the keywords, it routes the query to the corresponding agent (`FlightAgent`, `HotelAgent`, or `GuideAgent`).
5.  The selected agent processes the query using its predefined tools (which are currently mocked for flights and hotels) and returns a response.
6.  The response is displayed in the CLI.

## Future Development & How to Contribute

This project is a great starting point. Here’s how you can contribute to making it fully functional:

### 1. Connect the Flight and Hotel Agents to the Amadeus API

The `services/amadeus_client.py` file already contains a fully implemented client for the Amadeus API. The task is to integrate it into the agents.

-   **For the `FlightAgent`**:
    1.  Open `agents/flight_agent.py`.
    2.  In the `FlightSearchTool`, import the `AmadeusClient` and `FlightSearchParams`.
    3.  In the `_run` method, replace the mock data generation with calls to `amadeus_client.search_flights(params)`.
    4.  You will need to map the tool's input parameters to the `FlightSearchParams` data class.

-   **For the `HotelAgent`**:
    1.  Open `agents/hotel_agent.py`.
    2.  In the `HotelSearchTool`, import the `AmadeusClient` and `HotelSearchParams`.
    3.  In the `_run` method, replace the mock data generation with calls to `amadeus_client.search_hotels(params)`.
    4.  Map the tool's input parameters to the `HotelSearchParams` data class.

### 2. Expand the Guide Agent's Knowledge Base

The `GuideAgent`'s `DestinationInfoTool` currently uses a tiny, hardcoded list of cities. To make it more useful:

1.  You can add more `Document` objects to the `_create_default_knowledge_base` method in `agents/guide_agent.py`.
2.  A better approach would be to modify it to load documents from the `./data/knowledge_base` directory, which is what the original `README.md` intended. This would involve reading files (e.g., text or markdown files) from that directory and adding them to the Chroma vector store.

### 3. Improve Agent Routing

The current keyword-based router in `orchestrator.py` is simple. A more advanced implementation would use an LLM to decide which agent (or sequence of agents) is best suited to handle a complex query (e.g., "Plan a trip to Paris with a flight and hotel").

## Project File Structure

```
├── agents/
│   ├── base_agent.py        # Abstract base class for all agents
│   ├── flight_agent.py      # Mocked agent for flight queries
│   ├── hotel_agent.py       # Mocked agent for hotel queries
│   └── guide_agent.py       # Agent for destination info (partially RAG)
├── services/
│   └── amadeus_client.py    # (Currently Unused) Client for Amadeus API
├── .gitignore
├── cli.py                   # Main entry point for the interactive CLI
├── config.py                # (Currently Unused) Configuration file
├── orchestrator.py          # Coordinates the agents using LangGraph
├── README.md                # This file
└── requirements.txt         # Python dependencies
```
