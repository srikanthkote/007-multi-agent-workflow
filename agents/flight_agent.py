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
from services.amadeus_client import AmadeusClient, FlightSearchParams

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
        # In a real implementation, this would call a flight status API
        # This is a mock implementation
        import json
        from datetime import datetime
        
        try:
            params = json.loads(query)
            flight_number = params.get('flight_number', '').upper()
            date_str = params.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Generate a random status
            statuses = [
                'On Time', 'Delayed', 'Boarding', 'Departed', 'In Air',
                'Landed', 'Cancelled', 'Diverted'
            ]
            status = random.choice(statuses)
            
            # Generate a random gate
            gate = f"{random.choice(['A', 'B', 'C', 'D'])}{random.randint(1, 50)}"
            
            # Generate random departure/arrival times
            departure_time = f"{random.randint(6, 22):02d}:{random.choice(['00', '15', '30', '45'])}"
            arrival_time = f"{(int(departure_time[:2]) + random.randint(1, 6)) % 24:02d}:{random.choice(['00', '15', '30', '45'])}"
            
            return {
                'flight_number': flight_number,
                'date': date_str,
                'status': status,
                'departure_time': departure_time,
                'arrival_time': arrival_time,
                'gate': gate if status in ['Boarding', 'On Time', 'Delayed'] else None,
                'terminal': random.randint(1, 5) if gate else None
            }
            
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON input'}
    
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

# Example usage:
if __name__ == "__main__":
    import os
    from langchain_community.llms import HuggingFaceEndpoint
    import asyncio
    
    async def test_flight_agent():
        """Test the flight agent with mock responses"""
        from dotenv import load_dotenv
        from langchain_community.llms import FakeListLLM
        import json
        
        # Load environment variables
        load_dotenv()
        
        # Define mock responses for different queries
        mock_responses = {
            "Find me flights from New York to London on June 10, 2025": 
                json.dumps({
                    "action": "search_flights",
                    "flights": [
                        {"flight_number": "AA100", "departure": "08:00", "arrival": "20:00", "price": 1200, "airline": "American Airlines"},
                        {"flight_number": "BA200", "departure": "14:00", "arrival": "02:00", "price": 1100, "airline": "British Airways"},
                        {"flight_number": "DL300", "departure": "18:00", "arrival": "06:00", "price": 1150, "airline": "Delta"}
                    ]
                }),
            "What's the status of flight AA123?": 
                json.dumps({
                    "action": "flight_status",
                    "flight_number": "AA123",
                    "status": "On Time",
                    "departure": "08:00",
                    "arrival": "10:00",
                    "gate": "B12"
                })
        }
        
        # Create a custom FakeLLM that returns appropriate responses based on input
        class CustomFakeLLM(FakeListLLM):
            def _call(self, prompt: str, **kwargs):
                # Find the most appropriate response based on the input
                for query, response in mock_responses.items():
                    if query.lower() in prompt.lower():
                        return response
                return "I'm sorry, I couldn't process that request."
        
        # Initialize the LLM with our custom responses
        llm = CustomFakeLLM(responses=list(mock_responses.values()))
        
        # Create and initialize the flight agent
        print("Initializing Flight Agent...")
        agent = FlightAgent(llm=llm)
        
        def print_section(title):
            print(f"\n{'='*60}\n{title.upper():^60}\n{'='*60}")
        
        def print_response(response):
            if not response.success:
                print(f"❌ Error: {response.error}")
                return
                
            try:
                #data = json.loads(response.data.get('output', '{}'))
                data = response.data.get('response')
                flightResponse = json.loads(data)

                if flightResponse.get('action') == 'search_flights':
                    print("\n✈️ Available Flights:")
                    print("-" * 50)
                    for flight in flightResponse.get('flights', []):
                        print(f"{flight['airline']} {flight['flight_number']}")
                        print(f"  🕒 {flight['departure']} - {flight['arrival']}")
                        print(f"  💵 ${flight['price']}")
                        print()
                elif flightResponse.get('action') == 'flight_status':
                    print(f"\nℹ️ Flight {flightResponse['flight_number']} Status:")
                    print("-" * 50)
                    print(f"Status: {flightResponse['status']}")
                    print(f"Departure: {flightResponse['departure']}")
                    print(f"Arrival: {flightResponse['arrival']}")
                    print(f"Gate: {flightResponse['gate']}")
                else:
                    print(response.data.get('output', 'No data available'))
            except (json.JSONDecodeError, AttributeError):
                print(response.data.get('output', 'No data available'))
            except Exception as e:
                print(f"\n❌ Test failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
            
        try:
            # Test 1: Flight Search
            print_section("Test 1: Flight Search")
            search_query = "Find me flights from New York to London on June 10, 2025"
            print(f"Query: {search_query}")
            
            search_response = await agent.process_query(search_query)
            print_response(search_response)
            
            # Test 2: Flight Status
            print_section("Test 2: Flight Status")
            status_query = "What's the status of flight AA123?"
            print(f"Query: {status_query}")
            
            status_response = await agent.process_query(status_query)
            print_response(status_response)
            
            print_section("Test Complete")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_flight_agent())
