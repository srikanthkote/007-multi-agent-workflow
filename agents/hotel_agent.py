from typing import Dict, Any, List, Optional, Type
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .base_agent import BaseAgent, AgentResponse, BaseToolResponse

class HotelSearchTool(BaseTool):
    """Tool for searching hotels based on location and criteria"""
    name: str = "search_hotels"
    description: str = """Useful for finding hotels in a specific location.
    
    Input should be a JSON string with the following keys:
    - location: str (required) - The city or area to search in
    - check_in: str (optional) - Check-in date in YYYY-MM-DD format
    - check_out: str (optional) - Check-out date in YYYY-MM-DD format
    - guests: int (optional) - Number of guests
    - budget: str (optional) - Budget range (e.g., '100-200', 'luxury')
    - amenities: List[str] (optional) - Desired amenities (e.g., ['pool', 'gym', 'spa'])
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        # In a real implementation, this would call a hotel API
        # This is a mock implementation
        import json
        try:
            params = json.loads(query)
            location = params.get('location', 'unknown')
            
            # Mock response - in reality, this would be an API call
            mock_hotels = [
                {
                    'name': f'Grand {location} Hotel',
                    'price': 250,
                    'rating': 4.5,
                    'amenities': ['pool', 'gym', 'spa', 'restaurant'],
                    'available': True
                },
                {
                    'name': f'{location} Plaza',
                    'price': 180,
                    'rating': 4.2,
                    'amenities': ['gym', 'restaurant'],
                    'available': True
                },
                {
                    'name': f'Budget {location} Inn',
                    'price': 90,
                    'rating': 3.7,
                    'amenities': ['free_wifi'],
                    'available': True
                }
            ]
            
            # Filter by budget if specified
            if 'budget' in params:
                budget = params['budget']
                if budget == 'luxury':
                    mock_hotels = [h for h in mock_hotels if h['price'] > 200]
                elif '-' in budget:
                    min_price, max_price = map(int, budget.split('-'))
                    mock_hotels = [h for h in mock_hotels if min_price <= h['price'] <= max_price]
            
            # Filter by amenities if specified
            if 'amenities' in params and params['amenities']:
                required_amenities = set(params['amenities'])
                mock_hotels = [
                    h for h in mock_hotels 
                    if all(amenity in h['amenities'] for amenity in required_amenities)
                ]
            
            return {
                'hotels': mock_hotels[:5],  # Return top 5 results
                'search_parameters': params
            }
            
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON input'}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class HotelBookingTool(BaseTool):
    """Tool for making hotel reservations"""
    name: str = "book_hotel"
    description: str = """Useful for booking a hotel room.
    
    Input should be a JSON string with the following keys:
    - hotel_id: str (required) - ID of the hotel to book
    - check_in: str (required) - Check-in date in YYYY-MM-DD format
    - check_out: str (required) - Check-out date in YYYY-MM-DD format
    - guests: int (required) - Number of guests
    - rooms: int (optional, default=1) - Number of rooms
    - guest_name: str (required) - Name for the reservation
    - email: str (required) - Email for the reservation
    - special_requests: str (optional) - Any special requests
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        # In a real implementation, this would call a booking API
        # This is a mock implementation
        import json
        import random
        import string
        
        try:
            params = json.loads(query)
            
            # Generate a mock confirmation number
            confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            
            return {
                'confirmation_number': confirmation,
                'status': 'confirmed',
                'booking_details': params,
                'message': 'Your booking has been confirmed!',
                'cancellation_policy': 'Free cancellation up to 24 hours before check-in.'
            }
            
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON input'}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class HotelAgent(BaseAgent):
    """Agent specialized in handling hotel-related queries and bookings"""
    
    def __init__(
        self, 
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[VectorStore] = None,
        tools: Optional[List[BaseTool]] = None,
        name: str = "HotelAgent"
    ):
        # Initialize with default tools if none provided
        if tools is None:
            tools = [
                HotelSearchTool(),
                HotelBookingTool()
            ]
            
        super().__init__(
            name=name,
            llm=llm,
            config=config or {},
            vector_store=vector_store,
            tools=tools
        )
    
    def initialize_agent(self) -> None:
        """Initialize the hotel agent with tools and workflow"""
        # Create the agent executor
        self.agent_executor = self._create_agent_executor()
        
        # Create the workflow
        self.workflow = self._create_workflow()
        
        # Set up the system message
        self.system_message = SystemMessage(
            content=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return (
            "You are a helpful hotel booking assistant. Your role is to help users find "
            "and book hotels based on their preferences. Be friendly, professional, and thorough in your "
            "responses. Always confirm all details before finalizing a booking.\n\n"
            "When searching for hotels, make sure to ask about:\n"
            "- Location (city, area, or landmark)\n"
            "- Check-in and check-out dates\n"
            "- Number of guests and rooms\n"
            "- Budget range\n"
            "- Desired amenities\n"
            "When a user wants to book, collect all necessary information including:\n"
            "- Full name\n"
            "- Email address\n"
            "- Any special requests\n"
            "Always confirm all details before finalizing a booking."
        )
    
    async def process_query(self, query: str, context: Optional[Dict] = None) -> AgentResponse:
        try:
            # Prepare the input for the agent
            chat_history = context.get('chat_history', []) if context else []
            
            # Add system message if not already present
            if not any(isinstance(msg, SystemMessage) for msg in chat_history):
                system_message = SystemMessage(content=self._get_system_prompt())
                chat_history = [system_message] + chat_history
            
            # Add user message to chat history
            chat_history.append(HumanMessage(content=query))
            
            # Prepare the input state for the workflow
            input_state = {
                "input": query,
                "chat_history": chat_history,
                "agent_scratchpad": "",
                "agent_outcome": None
            }
            
            # Run the workflow
            result = await self.workflow.ainvoke(input_state)
            
            # Debug: Print the raw workflow result
            print("\nWorkflow result:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
            # Extract the output from the workflow result
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
            
        except Exception as e:
            self.logger.error(f"Error processing hotel query: {str(e)}", exc_info=True)
            return self._format_response(
                success=False,
                error=f"Error processing hotel query: {str(e)}"
            )

# Example usage:
if __name__ == "__main__":
    from langchain_community.llms import FakeListLLM
    import asyncio
    import json
    from datetime import datetime, timedelta
    
    async def test_hotel_agent():
        """Test the hotel agent with mock responses"""
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Generate check-in and check-out dates
        today = datetime.now()
        check_in = (today + timedelta(days=14)).strftime('%Y-%m-%d')  # 2 weeks from now
        check_out = (today + timedelta(days=21)).strftime('%Y-%m-%d')  # 1 week stay
        
        # Define mock responses for different queries
        mock_responses = {
            f"Find me a hotel in Paris for 2 adults from {check_in} to {check_out}": 
                json.dumps({
                    "action": "search_hotels",
                    "hotels": [
                        {
                            "name": "Grand Paris Hotel",
                            "price": 250,
                            "rating": 4.5,
                            "amenities": ["pool", "gym", "spa", "restaurant"],
                            "location": "Paris, France",
                            "available": True
                        },
                        {
                            "name": "Eiffel View Suites",
                            "price": 320,
                            "rating": 4.7,
                            "amenities": ["eiffel_tower_view", "restaurant", "bar", "concierge"],
                            "location": "Paris, France",
                            "available": True
                        },
                        {
                            "name": "Budget Paris Inn",
                            "price": 95,
                            "rating": 3.8,
                            "amenities": ["free_wifi", "breakfast"],
                            "location": "Paris, France",
                            "available": True
                        }
                    ]
                }),
            f"Book a room at the Grand Paris Hotel for 2 adults from {check_in} to {check_out}":
                json.dumps({
                    "action": "book_hotel",
                    "confirmation_number": "HOTEL-789012",
                    "hotel_name": "Grand Paris Hotel",
                    "check_in": check_in,
                    "check_out": check_out,
                    "guests": 2,
                    "total_price": 1750,
                    "status": "confirmed"
                })
        }
        
        # Create a custom FakeLLM that returns appropriate responses based on input
        class CustomFakeLLM(FakeListLLM):
            def _call(self, prompt: str, **kwargs):
                # Find the most appropriate response based on the input
                for query, response in mock_responses.items():
                    if query.lower() in prompt.lower():
                        return response
                return "I'm sorry, I couldn't process that hotel request."
        
        # Initialize the LLM with our custom responses
        llm = CustomFakeLLM(responses=list(mock_responses.values()))
        
        # Create and initialize the hotel agent
        print("Initializing Hotel Agent...")
        agent = HotelAgent(llm=llm)
        
        def print_section(title):
            print(f"\n{'='*60}\n{title.upper():^60}\n{'='*60}")
        
        def print_response(response, is_booking=False):
            if not response.success:
                print(f"❌ Error: {response.error}")
                return
                
            try:
                data = response.data
                if not isinstance(data, dict):
                    print(data)
                    return
                    
                if is_booking:
                    print("\n✅ Hotel Booking Confirmed!")
                    print("-" * 50)
                    print(f"Confirmation #: {data.get('confirmation_number', 'N/A')}")
                    print(f"Hotel: {data.get('hotel_name', 'N/A')}")
                    print(f"Check-in: {data.get('check_in', 'N/A')}")
                    print(f"Check-out: {data.get('check_out', 'N/A')}")
                    print(f"Guests: {data.get('guests', 'N/A')}")
                    print(f"Total Price: ${data.get('total_price', 'N/A')}")
                    print(f"Status: {data.get('status', 'N/A')}")
                else:
                    hotels = data.get('hotels', [])
                    if not hotels:
                        print("No hotels found matching your criteria.")
                        return
                        
                    print(f"\n🏨 Found {len(hotels)} Hotels in {hotels[0].get('location', 'your destination')}")
                    print("-" * 50)
                    
                    for i, hotel in enumerate(hotels, 1):
                        print(f"\n{i}. {hotel.get('name', 'Unnamed Hotel')}")
                        print(f"   ⭐ {hotel.get('rating', 'N/A')}/5")
                        print(f"   💵 ${hotel.get('price', 'N/A')} per night")
                        print(f"   🛏️  Amenities: {', '.join(hotel.get('amenities', []))}")
                        
            except Exception as e:
                print(f"Error formatting response: {str(e)}")
                print("Raw response:", data)
        
        try:
            # Test 1: Hotel Search
            print_section("Test 1: Hotel Search")
            search_query = f"Find me a hotel in Paris for 2 adults from {check_in} to {check_out}"
            print(f"Query: {search_query}")
            
            search_response = await agent.process_query(search_query)
            print_response(search_response)
            
            # Test 2: Hotel Booking
            print_section("Test 2: Hotel Booking")
            booking_query = f"Book a room at the Grand Paris Hotel for 2 adults from {check_in} to {check_out}"
            print(f"Query: {booking_query}")
            
            booking_response = await agent.process_query(booking_query)
            print_response(booking_response, is_booking=True)
            
            print_section("Test Complete")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_hotel_agent())
