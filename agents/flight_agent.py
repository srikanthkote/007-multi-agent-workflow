import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import random
import string
from .base_agent import BaseAgent, AgentResponse
from services.amadeus_client import AmadeusClient, FlightSearchParams, FlightStatusParams
import asyncio
import re

# Configure logger
logger = logging.getLogger(__name__)

class FlightSearchTool(BaseTool):
    """Tool for searching flights based on criteria"""
    name: str = "search_flights"
    description: str = """Useful for finding flights between two locations.
    
    Input should be a JSON string with the following keys:
    - origin: str (required) - Departure city/airport code
    - destination: str (required) - Arrival city/airport code
    - departure_date: str (required) - Departure date in YYYY-MM-DD format
    - return_date: str (optional) - Return date in YYYY-MM-DD format for round trips
    - passengers: int (optional, default=1) - Number of passengers
    - cabin_class: str (optional) - Cabin class (economy, premium_economy, business, first)
    - direct_flights: bool (optional) - Whether to show only direct flights
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Performs a flight search using the Amadeus API."""
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input for flight search."}

        try:
            # Initialize the Amadeus client
            amadeus_client = AmadeusClient()

            # Map tool input to FlightSearchParams
            search_params = FlightSearchParams(
                origin=params.get('origin'),
                destination=params.get('destination'),
                departure_date=params.get('departure_date'),
                return_date=params.get('return_date'),
                adults=params.get('passengers', 1),  # Map passengers to adults
                travel_class=params.get('cabin_class', 'ECONOMY').upper(),
                non_stop=params.get('direct_flights', False)
            )

            # Perform the search
            logger.info("Searching for flights with parameters: %s", search_params)
            search_result = amadeus_client.search_flights(search_params)
            
            if "error" in search_result:
                logger.error("Amadeus API returned an error: %s", search_result["error"])
                return {"error": f"Failed to retrieve flight data: {search_result['error']}"}

            return search_result

        except ValueError as ve:
            # This can happen if Amadeus API keys are not set
            logger.error("Configuration error for Amadeus client: %s", ve)
            return {"error": f"Configuration error: {ve}"}
        except Exception as e:
            logger.error("An unexpected error occurred during flight search: %s", e, exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}
        
    async def _arun(self, query: str) -> Dict[str, Any]:
        logger.info("Searching for : %s", query)
        # Async version of _run
        return self._run(query)

class FlightBookingTool(BaseTool):
    """Tool for booking flights"""
    name: str = "book_flight"
    description: str = """Useful for booking a flight.
    
    Input should be a JSON string with the following keys:
    - flight_id: str (required) - ID of the flight to book
    - passengers: List[Dict] (required) - List of passenger details
      - first_name: str (required)
      - last_name: str (required)
      - date_of_birth: str (required) - YYYY-MM-DD format
      - passport_number: str (optional)
    - contact_info: Dict (required)
      - email: str (required)
      - phone: str (required)
    - seat_preference: str (optional) - Window, Aisle, etc.
    - meal_preference: str (optional) - Vegetarian, Vegan, etc.
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        # In a real implementation, this would call a booking API
        # This is a mock implementation
        import json
        
        try:
            params = json.loads(query)
            
            # Generate a mock booking reference
            booking_ref = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            
            return {
                'booking_reference': booking_ref,
                'status': 'confirmed',
                'booking_details': params,
                'message': 'Your flight has been booked successfully!',
                'e_ticket_number': ''.join(random.choices(string.digits, k=13)),
                'cancellation_policy': 'Changes and cancellations may be allowed with fees. See fare rules for details.'
            }
            
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON input'}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class FlightStatusTool(BaseTool):
    """Tool for checking flight status"""
    name: str = "check_flight_status"
    description: str = """Useful for checking the status of a flight.
    
    Input should be a JSON string with the following keys:
    - flight_number: str (required) - Flight number (e.g., DL123)
    - date: str (optional) - Date in YYYY-MM-DD format (defaults to today)
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Checks flight status using the Amadeus API."""
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input for flight status check."}

        flight_number = params.get('flight_number')
        date = params.get('date')

        if not flight_number or not date:
            return {"error": "Missing required parameters: flight_number and date."}

        # Use regex to split carrier code and flight number
        match = re.match(r"([A-Z]{2})([0-9]+)", flight_number.upper())
        if not match:
            return {"error": "Invalid flight number format. Expected format like 'AA123'."}

        carrier_code, f_number = match.groups()

        try:
            amadeus_client = AmadeusClient()
            status_params = FlightStatusParams(
                carrier_code=carrier_code,
                flight_number=f_number,
                date=date
            )
            
            logger.info("Checking flight status with parameters: %s", status_params)
            status_result = amadeus_client.get_flight_status(status_params)
            
            if "error" in status_result:
                logger.error("Amadeus API returned an error: %s", status_result["error"])
                return {"error": f"Failed to retrieve flight status: {status_result['error']}"}

            return status_result

        except ValueError as ve:
            logger.error("Configuration error for Amadeus client: %s", ve)
            return {"error": f"Configuration error: {ve}"}
        except Exception as e:
            logger.error("An unexpected error occurred during flight status check: %s", e, exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class FlightAgent(BaseAgent):
    """Agent specialized in handling flight-related queries and bookings"""
    
    def __init__(
        self, 
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[VectorStore] = None,
        tools: Optional[List[BaseTool]] = None,
        name: str = "FlightAgent"
    ):
        # Initialize with default tools if none provided
        if tools is None:
            tools = [
                FlightSearchTool(),
                FlightBookingTool(),
                FlightStatusTool()
            ]
            
        super().__init__(
            name=name,
            llm=llm,
            config=config or {},
            vector_store=vector_store,
            tools=tools
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the flight agent.
        
        Returns:
            str: The system prompt
        """
        return """You are a helpful flight booking assistant. Your role is to help users find and 
        book flights based on their preferences. Be friendly, professional, and thorough in your 
        responses. Always confirm all details before finalizing a booking.

        When searching for flights, make sure to ask about:
        - Origin and destination cities/airports
        - Departure date (and return date for round trips)
        - Number of passengers
        - Cabin class preference
        - Any airline preferences or restrictions

        When a user wants to book a flight, collect all necessary information including:
        - Passenger details (full name, date of birth, passport info if international)
        - Contact information (email and phone number)
        - Seat and meal preferences if any

        Always confirm all details before finalizing a booking."""

    def initialize_agent(self) -> None:
        """Initialize the flight agent with tools and workflow"""
        # Create the agent executor
        self.agent_executor = self._create_agent_executor()
        
        # Create the workflow
        self.workflow = self._create_workflow()
        
        # Set up the system message
        self.system_message = SystemMessage(
            content="""You are a helpful flight booking assistant. Your role is to help users find and 
            book flights based on their preferences. Be friendly, professional, and thorough in your 
            responses. Always confirm all details before finalizing a booking.
            
            When searching for flights, make sure to ask about:
            - Origin and destination cities/airports
            - Departure date (and return date for round trips)
            - Number of passengers
            - Cabin class preference
            - Any airline preferences or restrictions
            
            When a user wants to book a flight, collect all necessary information including:
            - Passenger details (full name, date of birth, passport info if international)
            - Contact information (email and phone number)
            - Seat and meal preferences if any
            
            Always confirm all details before finalizing a booking."""
        )
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            if not self.workflow:
                return self._format_response(
                    success=False,
                    error="Flight agent not properly initialized"
                )
            
            # Prepare the input state with chat history
            chat_history = context.get('chat_history', []) if context else []
            
            # Add system message if not already present
            if not any(isinstance(msg, SystemMessage) for msg in chat_history):
                system_message = SystemMessage(content=self._get_system_prompt())
                chat_history = [system_message] + chat_history
                
            # Add user message
            chat_history.append(HumanMessage(content=query))
            
            # Prepare the input state
            input_state = {
                'input': query,
                'chat_history': chat_history,
                'agent_scratchpad': ''
            }
            
            try:
                result = await self.workflow.ainvoke(input_state)
                
                # Debug: Print the full result structure
                print("\nWorkflow result:")
                print("-" * 50)
                print(result)
                
                # Extract the final response
                if isinstance(result, dict):
                    # Case 1: Direct response in agent_outcome
                    if 'agent_outcome' in result and isinstance(result['agent_outcome'], dict):
                        if 'output' in result['agent_outcome']:
                            output = result['agent_outcome']['output']
                            return self._format_response(
                                success=True,
                                data={'output': output}
                            )
                    # Case 2: Response is in the root of the result
                    elif 'output' in result:
                        return self._format_response(
                            success=True,
                            data={'output': result['output']}
                        )
                
                return self._format_response(
                    success=False,
                    error="Unexpected response format from the workflow"
                )
                
            except asyncio.TimeoutError:
                return self._format_response(
                    success=False,
                    error="Request timed out. Please try again."
                )
            except Exception as e:
                logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
                return self._format_response(
                    success=False,
                    error=f"An error occurred while processing your request: {str(e)}"
                )
            
        except Exception as e:
            logger.error(f"Unexpected error in process_query: {str(e)}", exc_info=True)
            return self._format_response(
                success=False,
                error="An unexpected error occurred. Please try again later."
            )

