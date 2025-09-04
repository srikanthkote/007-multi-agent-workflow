"""
Amadeus API client for flight and hotel search functionality.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pytz
from amadeus import Client, ResponseError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class FlightSearchParams:
    """Parameters for flight search."""
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    travel_class: str = 'ECONOMY'
    non_stop: bool = False

@dataclass
class HotelSearchParams:
    """Parameters for hotel search."""
    city_code: str
    check_in: str
    check_out: str
    adults: int = 1
    room_quantity: int = 1
    price_range: Optional[str] = None
    rating: Optional[int] = None

@dataclass
class FlightStatusParams:
    """Parameters for flight status check."""
    carrier_code: str
    flight_number: str
    date: str

class AmadeusClient:
    """Client for interacting with Amadeus Self-Service APIs."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize the Amadeus client.
        
        Args:
            api_key: Amadeus API key (default: from environment)
            api_secret: Amadeus API secret (default: from environment)
        """
        self.api_key = api_key or os.getenv('AMADEUS_API_KEY')
        self.api_secret = api_secret or os.getenv('AMADEUS_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Amadeus API key and secret must be provided either "
                "through constructor or environment variables (AMADEUS_API_KEY, AMADEUS_API_SECRET)"
            )
        
        self.client = Client(
            client_id=self.api_key,
            client_secret=self.api_secret,
            log_level='debug' if os.getenv('DEBUG') else 'warn'
        )
    
    def search_flights(self, params: FlightSearchParams) -> Dict[str, Any]:
        """Search for flights using Amadeus Flight Offers Search API.
        
        Args:
            params: Flight search parameters
            
        Returns:
            Dictionary containing flight offers
        """
        try:
            # Prepare request parameters
            request_params = {
                'originLocationCode': params.origin.upper(),
                'destinationLocationCode': params.destination.upper(),
                'departureDate': params.departure_date,
                'adults': params.adults,
                'children': params.children,
                'travelClass': params.travel_class.upper(),
                'nonStop': "false",
                'max': 10  # Limit number of results
            }
            
            if params.return_date:
                request_params['returnDate'] = params.return_date
            
            # Make API request
            response = self.client.shopping.flight_offers_search.get(**request_params)
            return self._parse_flight_offers(response.data)
            
        except ResponseError as error:
            logger.error(f"Amadeus API error: {error}")
            return {"error": str(error), "status_code": error.status_code}
        except Exception as e:
            logger.error(f"Error searching flights: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def search_hotels(self, params: HotelSearchParams) -> Dict[str, Any]:
        """Search for hotels using Amadeus Hotel Search API.
        
        Args:
            params: Hotel search parameters
            
        Returns:
            Dictionary containing hotel offers
        """
        try:
            # First, get hotel list by city
            hotel_list = self._get_hotels_by_city(params.city_code)
            
            if not hotel_list or 'data' not in hotel_list or not hotel_list['data']:
                return {"error": "No hotels found in the specified location"}
            
            # Get hotel IDs for availability check
            hotel_ids = [hotel['hotelId'] for hotel in hotel_list['data'][:10]]  # Limit to first 10 hotels
            
            # Get availability and pricing
            availability = self._get_hotel_availability(
                hotel_ids=hotel_ids,
                check_in=params.check_in,
                check_out=params.check_out,
                adults=params.adults,
                room_quantity=params.room_quantity
            )
            
            return self._parse_hotel_offers(availability)
            
        except ResponseError as error:
            logger.error(f"Amadeus API error: {error}")
            return {"error": str(error), "status_code": error.status_code}
        except Exception as e:
            logger.error(f"Error searching hotels: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _get_hotels_by_city(self, city_code: str) -> Dict[str, Any]:
        """Get list of hotels in a city."""
        return self.client.reference_data.locations.hotels.by_city.get(
            cityCode=city_code.upper()
        )
    
    def _get_hotel_availability(self, hotel_ids: List[str], check_in: str, check_out: str, 
                              adults: int, room_quantity: int) -> Dict[str, Any]:
        """Get availability and pricing for specific hotels."""
        return self.client.shopping.hotel_offers_search.get(
            hotelIds=','.join(hotel_ids),
            checkInDate=check_in,
            checkOutDate=check_out,
            adults=adults,
            roomQuantity=room_quantity,
            bestRateOnly=True
        )
    
    def _parse_flight_offers(self, offers: List[Dict]) -> Dict[str, Any]:
        """Parse flight offers into a more readable format."""
        parsed_offers = []
        
        for offer in offers:
            itinerary = {
                'id': offer['id'],
                'price': {
                    'total': offer['price']['total'],
                    'currency': offer['price']['currency'],
                    'base': offer['price']['base']
                },
                'one_way': len(offer['itineraries']) == 1,
                'segments': []
            }
            
            # Process each itinerary segment
            for itinerary_segment in offer['itineraries']:
                segment = {
                    'duration': itinerary_segment['duration'],
                    'stops': len(itinerary_segment['segments']) - 1,
                    'segments': []
                }
                
                # Process each flight segment
                for flight_segment in itinerary_segment['segments']:
                    segment['segments'].append({
                        'departure': {
                            'airport': flight_segment['departure']['iataCode'],
                            'terminal': flight_segment.get('departure', {}).get('terminal'),
                            'time': flight_segment['departure']['at']
                        },
                        'arrival': {
                            'airport': flight_segment['arrival']['iataCode'],
                            'terminal': flight_segment.get('arrival', {}).get('terminal'),
                            'time': flight_segment['arrival']['at']
                        },
                        'carrier': flight_segment['carrierCode'],
                        'number': flight_segment['number'],
                        'aircraft': flight_segment.get('aircraft', {}).get('code'),
                        'duration': flight_segment['duration'],
                        'stops': len(flight_segment.get('stops', []))
                    })
                
                itinerary['segments'].append(segment)
            
            parsed_offers.append(itinerary)
        
        return {'data': parsed_offers}
    
    def _parse_hotel_offers(self, response: Dict) -> Dict[str, Any]:
        """Parse hotel offers into a more readable format."""
        if 'data' not in response or not response['data']:
            return {'data': []}
        
        parsed_offers = []
        
        for hotel in response['data']:
            offer = {
                'hotel_id': hotel['hotel']['hotelId'],
                'name': hotel['hotel']['name'],
                'rating': hotel['hotel'].get('rating', 'N/A'),
                'description': hotel['hotel'].get('description', {}).get('text', 'No description available'),
                'amenities': hotel['hotel'].get('amenities', []),
                'contact': {
                    'phone': hotel['hotel'].get('contact', {}).get('phone'),
                    'email': hotel['hotel'].get('contact', {}).get('email')
                },
                'address': hotel['hotel'].get('address', {}).get('lines', []),
                'city': hotel['hotel'].get('address', {}).get('cityName'),
                'postal_code': hotel['hotel'].get('address', {}).get('postalCode'),
                'country': hotel['hotel'].get('address', {}).get('countryCode'),
                'offers': []
            }
            
            # Process room offers
            for room_offer in hotel.get('offers', []):
                offer['offers'].append({
                    'id': room_offer['id'],
                    'room_type': room_offer.get('room', {}).get('typeEstimated', {}).get('bedType', 'Standard'),
                    'price': {
                        'total': room_offer['price']['total'],
                        'currency': room_offer['price']['currency']
                    },
                    'guests': room_offer.get('guests', {}).get('adults', 1),
                    'check_in': room_offer.get('checkInDate'),
                    'check_out': room_offer.get('checkOutDate'),
                    'cancellation_policy': room_offer.get('policies', {}).get('cancellation', {}).get('description')
                })
            
            parsed_offers.append(offer)
        
        return {'data': parsed_offers}

    def get_flight_status(self, params: FlightStatusParams) -> Dict[str, Any]:
        """Get flight status using Amadeus On-Demand Flight Status API."""
        try:
            response = self.client.schedule.flights.get(
                carrierCode=params.carrier_code.upper(),
                flightNumber=params.flight_number,
                scheduledDepartureDate=params.date
            )
            return self._parse_flight_status(response.data)
        except ResponseError as error:
            logger.error(f"Amadeus API error: {error}")
            return {"error": str(error), "status_code": error.status_code}
        except Exception as e:
            logger.error(f"Error getting flight status: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def _parse_flight_status(self, data: List[Dict]) -> Dict[str, Any]:
        """Parse flight status response into a more readable format."""
        if not data:
            return {"error": "Flight not found."}

        status_info = data[0]
        departure = status_info.get("departure", {})
        arrival = status_info.get("arrival", {})

        # Extract gate information from operational segments if available
        gate = None
        if "operationalFlightSegments" in status_info and status_info["operationalFlightSegments"]:
            gate = status_info["operationalFlightSegments"][0].get("departure", {}).get("gate")

        return {
            "flight_number": f"{status_info.get('carrierCode', '')}{status_info.get('flightNumber', '')}",
            "date": departure.get("scheduled", status_info.get("scheduledDepartureDate")),
            "status": status_info.get("status", "Unknown"),
            "departure_airport": departure.get("iataCode"),
            "departure_time": departure.get("at"),
            "departure_terminal": departure.get("terminal"),
            "arrival_airport": arrival.get("iataCode"),
            "arrival_time": arrival.get("at"),
            "arrival_terminal": arrival.get("terminal"),
            "gate": gate,
        }

# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize client
    amadeus = AmadeusClient()
    
    # Example flight search
    flight_params = FlightSearchParams(
        origin="NYC",
        destination="LON",
        departure_date="2024-10-15",
        return_date="2024-10-22",
        adults=1
    )
    flights = amadeus.search_flights(flight_params)
    print("Flight Search Results:")
    print(flights)
    
    # Example hotel search
    hotel_params = HotelSearchParams(
        city_code="LON",
        check_in="2024-10-15",
        check_out="2024-10-22",
        adults=2
    )
    hotels = amadeus.search_hotels(hotel_params)
    print("\nHotel Search Results:")
    print(hotels)
