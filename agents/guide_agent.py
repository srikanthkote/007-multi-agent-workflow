from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os
from pathlib import Path

from .base_agent import BaseAgent, AgentResponse

class DestinationInfoTool(BaseTool):
    """Tool for getting information about travel destinations"""
    name: str = "get_destination_info"
    description: str = """Useful for getting detailed information about a specific travel destination.
    
    Input should be a JSON string with the following keys:
    - destination: str (required) - The name of the destination (city, country, or landmark)
    - info_type: str (optional) - Type of information needed (e.g., 'overview', 'attractions', 'food', 'culture', 'safety')
    - language: str (optional) - Preferred language for the response (default: 'en')
    """
    
    def __init__(self, knowledge_base: VectorStore = None, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base or self._create_default_knowledge_base()
    
    def _create_default_knowledge_base(self) -> VectorStore:
        """Create a default knowledge base with sample travel information"""
        # Sample destination data (in a real app, this would come from a database or API)
        destinations = [
            {
                "name": "Paris, France",
                "type": "city",
                "country": "France",
                "continent": "Europe",
                "overview": "Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.",
                "attractions": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Champs-Élysées", "Montmartre"],
                "best_time_to_visit": "April to June and October to early November",
                "language": "French",
                "currency": "Euro (€)",
                "safety": "Generally safe, but beware of pickpockets in tourist areas.",
                "food": ["Croissants", "Baguettes", "Macarons", "Escargot", "Coq au Vin"]
            },
            {
                "name": "Tokyo, Japan",
                "type": "city",
                "country": "Japan",
                "continent": "Asia",
                "overview": "Tokyo, Japan's busy capital, mixes the ultramodern and the traditional, from neon-lit skyscrapers to historic temples.",
                "attractions": ["Shibuya Crossing", "Tokyo Skytree", "Senso-ji Temple", "Meiji Shrine", "Tsukiji Outer Market"],
                "best_time_to_visit": "March to April (cherry blossom season) and September to November",
                "language": "Japanese",
                "currency": "Yen (¥)",
                "safety": "Extremely safe with very low crime rates.",
                "food": ["Sushi", "Ramen", "Tempura", "Okonomiyaki", "Takoyaki"]
            },
            {
                "name": "New York City, USA",
                "type": "city",
                "country": "United States",
                "continent": "North America",
                "overview": "New York City comprises 5 boroughs sitting where the Hudson River meets the Atlantic Ocean. At its core is Manhattan, a densely populated borough that's among the world's major commercial, financial and cultural centers.",
                "attractions": ["Statue of Liberty", "Central Park", "Times Square", "Empire State Building", "Metropolitan Museum of Art"],
                "best_time_to_visit": "April to June and September to early November",
                "language": "English",
                "currency": "US Dollar ($)",
                "safety": "Generally safe in tourist areas, but be cautious in less crowded areas at night.",
                "food": ["Pizza", "Bagels", "Cheesecake", "Hot Dogs", "Pastrami on Rye"]
            }
        ]
        
        # Convert to documents for the vector store
        docs = []
        for dest in destinations:
            doc = Document(
                page_content=json.dumps(dest),
                metadata={
                    "name": dest["name"],
                    "type": dest["type"],
                    "country": dest["country"],
                    "continent": dest["continent"]
                }
            )
            docs.append(doc)
        
        # Create and return the vector store
        embeddings = HuggingFaceEmbeddings()
        return Chroma.from_documents(docs, embeddings, collection_name="travel_guide_kb")
    
    def _run(self, query: str) -> Dict[str, Any]:
        try:
            params = json.loads(query)
            destination = params.get('destination', '').strip()
            info_type = params.get('info_type', 'overview').lower()
            
            if not destination:
                return {"error": "Destination is required"}
            
            # Search the knowledge base for matching destinations
            results = self.knowledge_base.similarity_search(destination, k=1)
            
            if not results:
                return {"error": f"No information found for {destination}"}
            
            # Get the most relevant destination
            dest_data = json.loads(results[0].page_content)
            
            # Return the requested information type if available
            if info_type in dest_data:
                if isinstance(dest_data[info_type], list):
                    return {
                        "destination": dest_data["name"],
                        "info_type": info_type,
                        "data": dest_data[info_type]
                    }
                else:
                    return {
                        "destination": dest_data["name"],
                        "info_type": info_type,
                        "data": dest_data[info_type]
                    }
            else:
                # If specific info type not found, return overview
                return {
                    "destination": dest_data["name"],
                    "info_type": "overview",
                    "data": dest_data.get("overview", "No overview available.")
                }
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error retrieving destination info: {str(e)}"}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class LocalRecommendationsTool(BaseTool):
    """Tool for getting local recommendations (restaurants, activities, etc.)"""
    name: str = "get_local_recommendations"
    description: str = """Useful for getting local recommendations like restaurants, activities, and hidden gems.
    
    Input should be a JSON string with the following keys:
    - location: str (required) - The name of the destination
    - category: str (optional) - Type of recommendation (e.g., 'restaurants', 'activities', 'shopping', 'nightlife')
    - budget: str (optional) - Budget level ('budget', 'mid-range', 'luxury')
    - limit: int (optional) - Maximum number of recommendations to return (default: 5)
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        # In a real implementation, this would query a database or API
        # This is a mock implementation
        try:
            params = json.loads(query)
            location = params.get('location', '').strip()
            category = params.get('category', 'attractions').lower()
            budget = params.get('budget', 'mid-range')
            limit = min(int(params.get('limit', 5)), 10)  # Max 10 results
            
            if not location:
                return {"error": "Location is required"}
            
            # Mock data for different categories
            mock_data = {
                'restaurants': [
                    f"{budget.capitalize()} {cuisine} restaurant"
                    for cuisine in ['Italian', 'French', 'Japanese', 'Local', 'Fusion', 'Seafood', 'Vegetarian']
                ][:limit],
                'activities': [
                    f"{activity}"
                    for activity in [
                        'Walking tour', 'Cooking class', 'Bike tour', 'Museum visit',
                        'Local market tour', 'Hiking', 'Boat tour', 'Food tour'
                    ]
                ][:limit],
                'shopping': [
                    f"{item}"
                    for item in [
                        'Local artisan market', 'Shopping district', 'Boutique stores',
                        'Antique shops', 'Mall', 'Souvenir shops'
                    ]
                ][:limit],
                'nightlife': [
                    f"{venue}"
                    for venue in [
                        'Rooftop bar', 'Jazz club', 'Cocktail bar', 'Local pub',
                        'Nightclub', 'Wine bar', 'Live music venue'
                    ]
                ][:limit],
                'attractions': [
                    f"{attraction}"
                    for attraction in [
                        'Historic district', 'Famous landmark', 'Art gallery',
                        'Botanical garden', 'Observation deck', 'Local museum'
                    ]
                ][:limit]
            }
            
            # Get recommendations based on category
            if category in mock_data:
                recommendations = mock_data[category]
            else:
                recommendations = mock_data['attractions']
            
            return {
                "location": location,
                "category": category,
                "budget": budget,
                "recommendations": recommendations[:limit]
            }
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error getting recommendations: {str(e)}"}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class ItineraryPlannerTool(BaseTool):
    """Tool for creating travel itineraries"""
    name: str = "create_itinerary"
    description: str = """Useful for creating a travel itinerary for a destination.
    
    Input should be a JSON string with the following keys:
    - destination: str (required) - The name of the destination
    - days: int (required) - Number of days for the itinerary
    - interests: List[str] (optional) - List of interests (e.g., 'history', 'food', 'nature', 'shopping')
    - budget: str (optional) - Budget level ('budget', 'mid-range', 'luxury')
    - travel_style: str (optional) - Travel style (e.g., 'relaxed', 'fast-paced', 'family', 'solo')
    """
    
    def _run(self, query: str) -> Dict[str, Any]:
        # In a real implementation, this would generate a more detailed itinerary
        # This is a mock implementation
        try:
            params = json.loads(query)
            destination = params.get('destination', '').strip()
            days = int(params.get('days', 3))
            interests = params.get('interests', ['sightseeing'])
            budget = params.get('budget', 'mid-range')
            travel_style = params.get('travel_style', 'balanced')
            
            if not destination:
                return {"error": "Destination is required"}
            
            if days < 1:
                return {"error": "Number of days must be at least 1"}
            
            # Generate a sample itinerary
            itinerary = []
            for day in range(1, days + 1):
                day_plan = {
                    "day": day,
                    "morning": [
                        f"Breakfast at a local {budget} cafe",
                        f"Visit {destination}'s top attraction"
                    ],
                    "afternoon": [
                        f"Lunch at a {budget} local restaurant",
                        f"Explore {random.choice(['museums', 'parks', 'markets', 'historic sites'])}"
                    ],
                    "evening": [
                        f"Dinner at a {budget} restaurant",
                        f"{random.choice(['Evening walk', 'Sunset viewing', 'Cultural show'])} in {destination}"
                    ]
                }
                
                # Add activities based on interests
                if 'food' in interests:
                    day_plan["afternoon"].append("Food tour or cooking class")
                if 'history' in interests:
                    day_plan["morning"].append("Guided historical tour")
                if 'nature' in interests:
                    day_plan["afternoon"].append("Nature walk or hike")
                if 'shopping' in interests:
                    day_plan["afternoon"].append("Shopping at local markets")
                
                # Adjust based on travel style
                if 'relaxed' in travel_style.lower():
                    day_plan["afternoon"].append("Free time to relax")
                if 'fast' in travel_style.lower():
                    day_plan["morning"].append("Early start to maximize the day")
                
                itinerary.append(day_plan)
            
            return {
                "destination": destination,
                "days": days,
                "travel_style": travel_style,
                "budget": budget,
                "interests": interests,
                "itinerary": itinerary
            }
            
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
        except Exception as e:
            return {"error": f"Error creating itinerary: {str(e)}"}
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        # Async version of _run
        return self._run(query)

class GuideAgent(BaseAgent):
    """Agent specialized in providing travel information and recommendations"""
    
    def __init__(
        self, 
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[VectorStore] = None,
        tools: Optional[List[BaseTool]] = None,
        knowledge_base_path: Optional[Union[str, Path]] = None,
        name: str = "GuideAgent"
    ):
        # Initialize with default tools if none provided
        if tools is None:
            # Create the knowledge base
            knowledge_base = None
            if knowledge_base_path and os.path.exists(knowledge_base_path):
                try:
                    embeddings = HuggingFaceEmbeddings()
                    knowledge_base = Chroma(
                        persist_directory=str(knowledge_base_path),
                        embedding_function=embeddings
                    )
                except Exception as e:
                    print(f"Warning: Could not load knowledge base from {knowledge_base_path}: {str(e)}")
            
            tools = [
                DestinationInfoTool(knowledge_base=knowledge_base),
                LocalRecommendationsTool(),
                ItineraryPlannerTool()
            ]
            
        super().__init__(
            name=name,
            llm=llm,
            config=config or {},
            vector_store=vector_store,
            tools=tools
        )
    
    def initialize_agent(self) -> None:
        """Initialize the guide agent with tools and workflow"""
        # Create the agent executor
        self.agent_executor = self._create_agent_executor()
        
        # Create the workflow
        self.workflow = self._create_workflow()
        
        # Set up the system message
        self.system_message = SystemMessage(
            content="""You are a knowledgeable and friendly travel guide. Your role is to provide 
            helpful and accurate information about travel destinations, make personalized 
            recommendations, and assist with travel planning.
            
            When helping users:
            1. First understand their destination and travel preferences
            2. Provide relevant information about the destination (culture, weather, etc.)
            3. Suggest activities and attractions based on their interests
            4. Help create customized itineraries when requested
            5. Offer practical tips (local customs, transportation, safety, etc.)
            
            Be concise but thorough in your responses, and always verify information when possible."""
        )
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a travel guide query"""
        try:
            if not self.workflow:
                return self._format_response(
                    success=False,
                    error="Guide agent not properly initialized"
                )
            
            # Prepare the input state with chat history
            chat_history = context.get('chat_history', []) if context else []
            
            # Add system message if this is the first message
            if not any(isinstance(msg, SystemMessage) for msg in chat_history):
                chat_history = [self.system_message] + chat_history
            
            # Add the new user message
            chat_history.append(HumanMessage(content=query))
            
            # Prepare the input state
            input_state = {
                "input": query,
                "chat_history": chat_history
            }
            
            # Execute the workflow
            result = await self.workflow.ainvoke(input_state)
            
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
            self.logger.error(f"Error processing guide query: {str(e)}", exc_info=True)
            return self._format_response(
                success=False,
                error=f"Error processing guide query: {str(e)}"
            )

# Example usage:
if __name__ == "__main__":
    from langchain_community.llms import FakeListLLM
    import asyncio
    import json
    
    async def test_guide_agent():
        """Test the guide agent with mock responses"""
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Define mock responses for different types of queries
        mock_responses = {
            "tell me about paris": json.dumps({
                "destination": "Paris, France",
                "overview": "Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine.",
                "top_attractions": [
                    "Eiffel Tower",
                    "Louvre Museum",
                    "Notre-Dame Cathedral",
                    "Champs-Élysées",
                    "Montmartre"
                ],
                "best_time_to_visit": "April to June and September to early November",
                "local_currency": "Euro (€)",
                "language": "French"
            }),
            "restaurants in tokyo": json.dumps({
                "location": "Tokyo",
                "category": "restaurants",
                "recommendations": [
                    {
                        "name": "Sukiyabashi Jiro",
                        "type": "Sushi",
                        "rating": 4.8,
                        "price_range": "$$$$",
                        "address": "4-2-15 Ginza, Chuo-ku, Tokyo",
                        "description": "World-famous sushi restaurant with three Michelin stars, featured in the documentary 'Jiro Dreams of Sushi'."
                    },
                    {
                        "name": "Ishikawa",
                        "type": "Kaiseki",
                        "rating": 4.7,
                        "price_range": "$$$",
                        "address": "5-37 Kagurazaka, Shinjuku-ku, Tokyo",
                        "description": "Michelin-starred restaurant serving traditional multi-course Japanese kaiseki cuisine in a beautiful setting."
                    }
                ]
            }),
            "3 day itinerary for new york": json.dumps({
                "destination": "New York City",
                "days": 3,
                "itinerary": [
                    {
                        "day": 1,
                        "title": "Iconic Landmarks",
                        "activities": [
                            "Morning: Visit Times Square and Broadway",
                            "Afternoon: Explore Central Park and visit the MET",
                            "Evening: See the city from Top of the Rock"
                        ]
                    },
                    {
                        "day": 2,
                        "title": "Museums and Views",
                        "activities": [
                            "Morning: Visit the American Museum of Natural History",
                            "Afternoon: Walk across Brooklyn Bridge",
                            "Evening: Dinner in DUMBO with Manhattan skyline views"
                        ]
                    },
                    {
                        "day": 3,
                        "title": "Neighborhood Exploration",
                        "activities": [
                            "Morning: Walk through Greenwich Village and SoHo for shopping",
                            "Afternoon: Visit the High Line and Chelsea Market",
                            "Evening: Enjoy dinner in Little Italy and dessert in Chinatown"
                        ]
                    }
                ]
            })
        }
        
        # Create a list of responses in the order they should be returned
        responses = [
            mock_responses["tell me about paris"],
            mock_responses["restaurants in tokyo"],
            mock_responses["3 day itinerary for new york"]
        ]
        
        # Initialize the FakeListLLM with our responses
        llm = FakeListLLM(responses=responses)
        
        # Create and initialize the guide agent with mock tools
        print("Initializing Guide Agent...")
        
        # Create mock tools to avoid loading real ones
        class MockTool(BaseTool):
            name: str = "mock_tool"
            description: str = "A mock tool for testing"
            
            def _run(self, *args, **kwargs) -> str:
                return "Mock tool response"
                
            async def _arun(self, *args, **kwargs) -> str:
                return "Mock tool response"
        
        # Initialize the agent with mock tools
        agent = GuideAgent(
            llm=llm,
            tools=[MockTool()]
        )
        
        def print_section(title):
            print(f"\n{'='*60}\n{title.upper():^60}\n{'='*60}")
        
        def print_response(response, response_type=None):
            if not response.success:
                print(f"❌ Error: {response.error}")
                return
                
            try:
                # Get the output from the response
                output = response.data.get('output', '')
                
                # Try to parse the output as JSON
                try:
                    data = json.loads(output) if isinstance(output, str) else output
                    print("\n" + json.dumps(data, indent=2, ensure_ascii=False))
                except:
                    print("\nResponse:", output)
                    
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                print("Raw response:", response.data)
        
        try:
            # Test 1: Destination Information
            print_section("Test 1: Destination Information")
            info_query = "Tell me about Paris, France"
            print(f"Query: {info_query}")
            
            info_response = await agent.process_query(info_query)
            print_response(info_response, "info")
            
            # Test 2: Local Recommendations
            print_section("Test 2: Local Recommendations")
            recs_query = "What are some good restaurants in Tokyo?"
            print(f"Query: {recs_query}")
            
            recs_response = await agent.process_query(recs_query)
            print_response(recs_response, "recommendations")
            
            # Test 3: Itinerary Planning
            print_section("Test 3: Itinerary Planning")
            itinerary_query = "Plan a 3-day itinerary for New York City"
            print(f"Query: {itinerary_query}")
            
            itinerary_response = await agent.process_query(itinerary_query)
            print_response(itinerary_response, "itinerary")
            
            print_section("Test Complete")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the test
    asyncio.run(test_guide_agent())
