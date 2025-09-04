from typing import Dict, Any, List, Optional, Union, Type
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
import logging

from agents.hotel_agent import HotelAgent
from agents.flight_agent import FlightAgent, FlightSearchTool, FlightBookingTool, FlightStatusTool
from agents.guide_agent import GuideAgent
from agents.base_agent import AgentResponse, AgentState

class TravelOrchestrator:
    """
    Orchestrates interactions between different travel-related agents
    to provide a comprehensive travel planning experience.
    """
    
    def __init__(
        self,
        llm: Union[BaseChatModel, BaseLLM],
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[VectorStore] = None,
        knowledge_base_path: Optional[str] = None
    ):
        self.llm = llm
        self.config = config or {}
        self.vector_store = vector_store
        self.knowledge_base_path = knowledge_base_path
        self.logger = self._setup_logger()
        
        # Initialize agents with empty tools list to start
        self.flight_agent = FlightAgent(
            name="FlightAgent",
            llm=llm, 
            vector_store=vector_store,
            #tools=[FlightSearchTool(), FlightBookingTool(), FlightStatusTool()]
            tools=[FlightSearchTool()]
        )
        self.hotel_agent = HotelAgent(
            name="HotelAgent",
            llm=llm, 
            vector_store=vector_store,
            tools=[]  # Initialize with empty tools list
        )
        self.guide_agent = GuideAgent(
            name="GuideAgent",
            llm=llm, 
            vector_store=vector_store,
            knowledge_base_path=knowledge_base_path,
            tools=[]  # Initialize with empty tools list
        )

        # Initialize the workflow
        self.workflow = self._create_workflow()
        
        # Set up the system message
        self.system_message = SystemMessage(
            content="""You are a helpful travel planning assistant. Your role is to coordinate between 
            specialized travel agents to help users plan their trips. You have access to the following 
            specialized agents:
            
            1. Hotel Agent - Handles hotel searches and bookings
            2. Flight Agent - Handles flight searches and bookings
            3. Guide Agent - Provides destination information and recommendations
            
            Based on the user's query, determine which agent(s) to use and coordinate their responses 
            to provide a comprehensive answer. Always be clear about which agent is providing which 
            information."""
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the orchestrator"""
        logger = logging.getLogger("TravelOrchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(log_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _create_workflow(self) -> StateGraph:
        """Create a LangGraph workflow for the orchestrator"""
        # Define the nodes
        workflow = StateGraph(state_schema=AgentState)
        
        # Add nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("flight_agent", self._route_to_flight_agent)
        workflow.add_node("hotel_agent", self._route_to_hotel_agent)        
        workflow.add_node("guide_agent", self._route_to_guide_agent)
        
        # Define the entry point
        workflow.set_entry_point("orchestrator")
        
        # Define the edges
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_to_agent,
            {
                "hotel": "hotel_agent",
                "flight": "flight_agent",
                "guide": "guide_agent",
                "multiple": "orchestrator",
                "end": END
            }
        )
        
        # Add edges back to orchestrator
        workflow.add_edge("flight_agent", "orchestrator")
        workflow.add_edge("hotel_agent", "orchestrator")
        workflow.add_edge("guide_agent", "orchestrator")
        
        # Compile the workflow
        graph = workflow.compile()        
        return graph
    
    async def _orchestrator_node(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """Node that determines which agent to route to"""
        #print("state:" + str(state))
        query = state.get("input", "")
        chat_history = state.get("chat_history", [])
        
        # If this is the first message, add the system message
        if not any(isinstance(msg, SystemMessage) for msg in chat_history):
            chat_history = [self.system_message] + chat_history
        
        # Add the user's message to the chat history
        chat_history.append(HumanMessage(content=query))
        
        # Determine which agent to route to
        route_decision = await self._determine_agent_route(query)
        return {
            "input": query,
            "chat_history": chat_history,
            "route_decision": route_decision,
            "agent_responses": state.get("agent_responses", {})
        }
    
    async def _determine_agent_route(self, query: str) -> str:
        print("_determine_agent_route")
        """Determine which agent should handle the query"""
        # Simple keyword-based routing - in a real app, you might use an LLM for this
        query_lower = query.lower()
        
        hotel_keywords = ['hotel', 'accommodation', 'stay', 'room', 'book a room', 'reservation']
        flight_keywords = ['flight', 'airline', 'fly', 'airport', 'book a flight', 'ticket']
        guide_keywords = ['things to do', 'attraction', 'restaurant', 'itinerary', 'plan', 'recommend', 'where to go', 'what to see']
        
        hotel_match = any(keyword in query_lower for keyword in hotel_keywords)
        flight_match = any(keyword in query_lower for keyword in flight_keywords)
        guide_match = any(keyword in query_lower for keyword in guide_keywords)
        
        # Count the number of matches
        matches = sum([hotel_match, flight_match, guide_match])
        print("query:matches: " + str(matches) + str(flight_match) + query_lower)
        if matches == 0:
            return "end"
        elif matches == 1:
            # If only one match, route to that agent
            if hotel_match:
                return "hotel"
            elif flight_match:
                return "flight"
            else:
                return "guide"
        else:
            # If multiple matches, we'll need to handle this in the orchestrator
            return "multiple"
    
    def _route_to_agent(self, state: Dict[str, Any]) -> str:
        """Determine the next node based on the route decision"""
        route_decision = state.get("route_decision", "end")
        
        if route_decision == "multiple":
            # For complex queries that might involve multiple agents
            return "orchestrator"

        print("_route_to_agent returns: " + str(route_decision))
        return route_decision
    
    async def _route_to_hotel_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("_route_to_hotel_agent")
        """Route the query to the hotel agent"""
        query = state["input"]
        chat_history = state["chat_history"]
        
        # Process with hotel agent
        response = await self.hotel_agent.process_query(
            query=query,
            context={"chat_history": chat_history}
        )
        
        agent_output = response.data.get('output')
        agent_responses = {"hotels": agent_output}
        # Add the agent's response to the chat history
        chat_history.append(AIMessage(content=agent_output))

        # Update agent responses
        #agent_responses = state.get("agent_responses", {})
        #agent_responses["hotel"] = response.data.get("output", "")
        
        # Add the agent's response to the chat history
        #chat_history.append(AIMessage(content=response.data.get("output", "")))
        
        return {
            "input": "",  # Clear input to avoid reprocessing
            "chat_history": chat_history,
            "agent_responses": agent_responses,
            "route_decision": "end"  # End after handling with one agent
        }
    
    async def _route_to_flight_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("_route_to_flight_agent")
        """Route the query to the flight agent"""
        query = state["input"]
        chat_history = state["chat_history"]
        
        # Process with flight agent
        response = await self.flight_agent.process_query(
            query=query,
            context={"chat_history": chat_history}
        )
        # Update agent responses
        agent_output = response.data.get('output')
        agent_responses = {"flight": agent_output}
        # Add the agent's response to the chat history
        chat_history.append(AIMessage(content=agent_output))
        
        return {
            "input": "",  # Clear input to avoid reprocessing
            "chat_history": chat_history,
            "agent_responses": agent_responses,
            "route_decision": "end"  # End after handling with one agent
        }
    
    async def _route_to_guide_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("_route_to_guide_agent")

        """Route the query to the guide agent"""
        query = state["input"]
        chat_history = state["chat_history"]
        
        # Process with guide agent
        response = await self.guide_agent.process_query(
            query=query,
            context={"chat_history": chat_history}
        )
        
        # Update agent responses
        agent_responses = state.get("agent_responses", {})
        agent_responses["guide"] = response.data.get("output", "")
        
        # Add the agent's response to the chat history
        chat_history.append(AIMessage(content=response.data.get("output", "")))
        
        return {
            "input": "",  # Clear input to avoid reprocessing
            "chat_history": chat_history,
            "agent_responses": agent_responses,
            "route_decision": "end"  # End after handling with one agent
        }
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a travel-related query by routing to the appropriate agent(s)"""
        try:
            if not query.strip():
                return AgentResponse(
                    success=False,
                    data={"output": "Please provide a valid query."},
                    error="Empty query"
                )
            
            # Prepare the initial state
            initial_state = AgentState({
                "input": query,
                "chat_history": context.get("chat_history", []) if context else [],
                "agent_responses": {}
            })
            
            # Execute the workflow

            result = await self.workflow.ainvoke(initial_state)
            #print("result: "+ str(result))
            # Get the final response
            chat_history = result.get("chat_history", [])            
            agent_responses = result.get("agent_responses", {})
            
            # Combine responses if multiple agents were involved
            if len(agent_responses) > 1:
                combined_response = "\n\n".join(
                    f"## {agent.capitalize()} Agent:\n{response}"
                    for agent, response in agent_responses.items()
                )
            else:
                combined_response = next(iter(agent_responses.values()), "I couldn't process that request.")
            
            return AgentResponse(
                success=True,
                data={
                    "output": combined_response,
                    "agent_responses": agent_responses
                },
                metadata={
                    "chat_history": chat_history
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return AgentResponse(
                success=False,
                data={"output": "An error occurred while processing your request."},
                error=str(e)
            )

# Example usage:
if __name__ == "__main__":
    from langchain_community.llms import HuggingFaceHub
    import asyncio
    
    async def test_orchestrator():
        # Initialize the language model
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.7, "max_length": 500}
        )
        
        # Create the orchestrator
        orchestrator = TravelOrchestrator(llm=llm)
        
        # Test queries
        queries = [
            "Find me a hotel in Paris for next weekend",
            "What are some must-see attractions in Tokyo?",
            "I need to book a flight from New York to London in July",
            "Plan a 3-day trip to Rome including flights and hotels"
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print(f"{'='*80}")
            
            response = await orchestrator.process_query(query)
            print("\nRESPONSE:")
            print(response.data['output'])
            print("\n" + "-"*40 + "\n")
    
    # Run the test
    asyncio.run(test_orchestrator())
