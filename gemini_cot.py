# enhanced_travel_planner.py
import os
import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from math import radians, cos, sin, asin, sqrt
from pydantic import BaseModel
import google.generativeai as genai
import googlemaps
from abc import ABC, abstractmethod


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Coordinates:
    lat: float
    lng: float


@dataclass
class EnhancedLocation:
    id: str
    name: str
    description: str
    tags: List[str]
    city: str
    area: str
    image: str
    coordinates: Coordinates
    city_id: int


@dataclass
class WeatherData:
    location: str
    date: str
    temperature: float
    condition: str
    humidity: int
    wind_speed: float
    precipitation_chance: int
    description: str
    forecast_days: List[Dict] = None


@dataclass
class RestaurantData:
    place_id: str
    name: str
    rating: float
    price_level: int
    cuisine_type: List[str]
    address: str
    coordinates: Coordinates
    opening_hours: Dict
    photos: List[str]


@dataclass
class HotelData:
    place_id: str
    name: str
    rating: float
    price_range: str
    amenities: List[str]
    address: str
    coordinates: Coordinates
    photos: List[str]
    room_types: List[str]


class UserPreferences(BaseModel):
    # Required fields
    city: str
    city_id: str
    main_tickets_booked: bool
    number_of_adults: int
    number_of_children: int
    budget_per_person: int
    local_transport: str
    hotel_quality: str
    pool_required: bool
    interests: List[str]
    
    # Optional fields
    start_datetime: Optional[str] = None
    start_location: Optional[str] = None
    end_datetime: Optional[str] = None
    end_location: Optional[str] = None
    dietary_restrictions: List[str] = []
    description_of_trip: Optional[str] = None


class SwipeDecision(BaseModel):
    """Represents a user's decision on a location."""
    locationId: str
    choice: str  # 'superlike', 'right', 'left'


@dataclass
class ChainOfThoughtReasoning:
    step: str
    reasoning: str
    data_gathered: Dict
    next_actions: List[str]


# =============================================================================
# ABSTRACT BASE CLASSES FOR TOOLS
# =============================================================================
class BaseAgent(ABC):
    """Abstract base class for all specialized agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.reasoning_chain = []
    
    @abstractmethod
    def execute(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main functionality"""
        pass
    
    def add_reasoning_step(self, step: str, reasoning: str, data: Dict, next_actions: List[str]):
        """Add a step to the chain of thought reasoning"""
        self.reasoning_chain.append(ChainOfThoughtReasoning(
            step=step,
            reasoning=reasoning,
            data_gathered=data,
            next_actions=next_actions
        ))


# =============================================================================
# SPECIALIZED TOOL AGENTS
# =============================================================================
class WeatherAgent(BaseAgent):
    """Specialized agent for weather-related queries with chain-of-thought reasoning"""
    
    def __init__(self, weather_key: str = None):
        super().__init__("WeatherAgent")
        self.weather_key = weather_key
    
    def execute(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute weather analysis with chain-of-thought reasoning
        Expected query format: {
            'location': str,
            'coordinates': {'lat': float, 'lng': float},
            'dates': ['start_date', 'end_date']
        }
        """
        self.reasoning_chain = []  # Reset reasoning chain
        
        # Step 1: Analyze the query
        self.add_reasoning_step(
            step="Query Analysis",
            reasoning="Analyzing the weather request to determine what specific weather information is needed",
            data={'query': query},
            next_actions=["fetch_current_weather", "fetch_forecast", "analyze_travel_suitability"]
        )
        
        location = query.get('location')
        coordinates = query.get('coordinates')
        dates = query.get('dates', [])
        
        if not coordinates:
            self.add_reasoning_step(
                step="Geocoding",
                reasoning="No coordinates provided, need to geocode the location first",
                data={'location': location},
                next_actions=["geocode_location"]
            )
            coordinates = self._geocode_location(location)
        
        # Step 2: Fetch current weather
        current_weather = self._fetch_current_weather(coordinates['lat'], coordinates['lng'])
        self.add_reasoning_step(
            step="Current Weather Retrieval",
            reasoning="Fetching current weather conditions to understand immediate climate",
            data=current_weather,
            next_actions=["fetch_forecast"]
        )
        
        # Step 3: Fetch weather forecast if dates provided
        forecast_data = None
        if dates:
            forecast_data = self._fetch_weather_forecast(coordinates['lat'], coordinates['lng'], dates)
            self.add_reasoning_step(
                step="Forecast Analysis",
                reasoning="Analyzing weather forecast for travel dates to provide travel recommendations",
                data={'forecast': forecast_data},
                next_actions=["generate_travel_weather_advice"]
            )
        
        # Step 4: Generate travel advice
        travel_advice = self._generate_weather_travel_advice(current_weather, forecast_data, dates)
        self.add_reasoning_step(
            step="Travel Weather Advisory",
            reasoning="Synthesizing weather data to provide actionable travel advice",
            data={'advice': travel_advice},
            next_actions=["return_comprehensive_weather_report"]
        )
        
        return {
            'current_weather': current_weather,
            'forecast': forecast_data,
            'travel_advice': travel_advice,
            'reasoning_chain': [
                {
                    'step': r.step,
                    'reasoning': r.reasoning,
                    'next_actions': r.next_actions
                } for r in self.reasoning_chain
            ]
        }
    
    def _geocode_location(self, location: str) -> Dict[str, float]:
        """Geocode location to coordinates (fallback implementation for WeatherAgent)"""
        try:
            # WeatherAgent doesn't have maps client, so provide a simple fallback
            # In practice, coordinates should be provided directly to WeatherAgent
            print(f"WeatherAgent: No geocoding available, using default coordinates for {location}")
            # Default to some coordinates (could be enhanced to use a simple geocoding service)
            return {'lat': 0, 'lng': 0}
        except Exception as e:
            print(f"Geocoding error: {e}")
        return {'lat': 0, 'lng': 0}
    
    def _fetch_current_weather(self, lat: float, lng: float) -> Dict[str, Any]:
        """Fetch current weather from OpenWeatherMap API"""
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={self.weather_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                'uv_index': None  # Current weather API doesn't include UV index
            }
        except Exception as e:
            print(f"Weather API error: {e}")
            return self._get_mock_weather()
    
    def _fetch_weather_forecast(self, lat: float, lng: float, dates: List[str]) -> List[Dict[str, Any]]:
        """Fetch weather forecast for specific dates"""
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&appid={self.weather_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            forecast_list = []
            for item in data['list'][:5]:  # Get 5-day forecast
                forecast_list.append({
                    'date': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'min_temp': item['main']['temp_min'],
                    'max_temp': item['main']['temp_max'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'],
                    'wind_speed': item['wind']['speed'],
                    'precipitation_probability': item.get('pop', 0) * 100
                })
            
            return forecast_list
        except Exception as e:
            print(f"Forecast API error: {e}")
            return self._get_mock_forecast()
    
    def _generate_weather_travel_advice(self, current_weather: Dict, forecast: List[Dict], dates: List[str]) -> Dict[str, Any]:
        """Generate travel advice based on weather data"""
        advice = {
            'clothing_recommendations': [],
            'activity_suggestions': [],
            'weather_warnings': [],
            'best_times_to_visit': []
        }
        
        # Analyze current temperature
        temp = current_weather.get('temperature', 20)
        if temp < 10:
            advice['clothing_recommendations'].append("Pack warm clothing, jackets, and layers")
        elif temp > 30:
            advice['clothing_recommendations'].append("Light, breathable clothing and sun protection")
        else:
            advice['clothing_recommendations'].append("Comfortable casual wear with light jacket")
        
        # Check for rain/precipitation
        if 'rain' in current_weather.get('description', '').lower():
            advice['clothing_recommendations'].append("Waterproof jacket and umbrella recommended")
            advice['activity_suggestions'].append("Indoor activities recommended during rain")
        
        # Analyze forecast if available
        if forecast:
            rainy_days = sum(1 for day in forecast if day.get('precipitation_probability', 0) > 50)
            if rainy_days > 2:
                advice['weather_warnings'].append("Multiple rainy days expected - pack accordingly")
            
            temps = [day.get('temperature', 20) for day in forecast]
            if max(temps) - min(temps) > 15:
                advice['weather_warnings'].append("Large temperature variations expected - pack layers")
        
        return advice
    
    def _get_mock_weather(self) -> Dict[str, Any]:
        """Fallback mock weather data"""
        return {
            'temperature': 22,
            'feels_like': 24,
            'humidity': 65,
            'pressure': 1013,
            'description': 'partly cloudy',
            'wind_speed': 3.5,
            'wind_direction': 180,
            'visibility': 10
        }
    
    def _get_mock_forecast(self) -> List[Dict[str, Any]]:
        """Fallback mock forecast data"""
        return [
            {
                'date': '2025-12-21 12:00:00',
                'temperature': 23,
                'min_temp': 18,
                'max_temp': 27,
                'humidity': 70,
                'description': 'sunny',
                'wind_speed': 4.0,
                'precipitation_probability': 10
            }
        ]


class RestaurantAgent(BaseAgent):
    """Specialized agent for restaurant searches with chain-of-thought reasoning"""
    
    def __init__(self, gmaps_key: str = None):
        super().__init__("RestaurantAgent")
        self.gmaps_key = gmaps_key
        # Initialize Google Maps client with the provided key
        if self.gmaps_key:
            self.gmaps_client = googlemaps.Client(key=self.gmaps_key)
        else:
            self.gmaps_client = None
    
    def execute(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute restaurant search with chain-of-thought reasoning
        Expected query format: {
            'location': str,
            'coordinates': {'lat': float, 'lng': float},
            'cuisine_preferences': List[str],
            'price_range': str,
            'dietary_restrictions': List[str],
            'radius': int (in meters)
        }
        """
        self.reasoning_chain = []
        
        # Step 1: Query Analysis
        self.add_reasoning_step(
            step="Restaurant Query Analysis",
            reasoning="Analyzing restaurant search requirements including cuisine, location, and user preferences",
            data={'query': query},
            next_actions=["determine_search_parameters", "fetch_restaurants"]
        )
        
        coordinates = query.get('coordinates')
        location = query.get('location')
        cuisine_prefs = query.get('cuisine_preferences', [])
        price_range = query.get('price_range', 'moderate')
        dietary_restrictions = query.get('dietary_restrictions', [])
        radius = query.get('radius', 2000)
        
        if not coordinates and location:
            coordinates = self._geocode_location(location)
        
        # Step 2: Determine search strategy
        search_strategy = self._determine_search_strategy(cuisine_prefs, price_range, dietary_restrictions)
        self.add_reasoning_step(
            step="Search Strategy Formulation",
            reasoning="Determining the best approach to find restaurants matching user preferences",
            data=search_strategy,
            next_actions=["execute_restaurant_search"]
        )
        
        # Step 3: Execute restaurant search
        restaurants = self._search_restaurants(coordinates, radius, search_strategy)
        self.add_reasoning_step(
            step="Restaurant Data Retrieval",
            reasoning="Fetching restaurant data from Google Places API based on search parameters",
            data={'found_restaurants': len(restaurants)},
            next_actions=["filter_and_rank_restaurants"]
        )
        
        # Step 4: Filter and rank results
        filtered_restaurants = self._filter_and_rank_restaurants(restaurants, query)
        self.add_reasoning_step(
            step="Restaurant Filtering and Ranking",
            reasoning="Applying filters and ranking restaurants based on user preferences and ratings",
            data={'final_restaurants': len(filtered_restaurants)},
            next_actions=["return_restaurant_recommendations"]
        )
        
        return {
            'restaurants': filtered_restaurants,
            'search_strategy': search_strategy,
            'total_found': len(restaurants),
            'reasoning_chain': [
                {
                    'step': r.step,
                    'reasoning': r.reasoning,
                    'next_actions': r.next_actions
                } for r in self.reasoning_chain
            ]
        }
    
    def _geocode_location(self, location: str) -> Dict[str, float]:
        """Geocode location to coordinates"""
        try:
            result = self.gmaps_client.geocode(location)
            if result:
                loc = result[0]['geometry']['location']
                return {'lat': loc['lat'], 'lng': loc['lng']}
        except Exception as e:
            print(f"Geocoding error: {e}")
        return {'lat': 0, 'lng': 0}
    
    def _determine_search_strategy(self, cuisine_prefs: List[str], price_range: str, dietary_restrictions: List[str]) -> Dict[str, Any]:
        """Determine the optimal search strategy for restaurants"""
        strategy = {
            'search_types': ['restaurant'],
            'keywords': [],
            'filters': [],
            'ranking_factors': ['rating', 'reviews', 'distance']
        }
        
        # Add cuisine-specific keywords
        if cuisine_prefs:
            strategy['keywords'].extend(cuisine_prefs)
        
        # Add dietary restriction filters
        if 'vegetarian' in dietary_restrictions:
            strategy['keywords'].append('vegetarian')
        if 'vegan' in dietary_restrictions:
            strategy['keywords'].append('vegan')
        if 'halal' in dietary_restrictions:
            strategy['keywords'].append('halal')
        
        # Price range considerations
        if price_range == 'budget':
            strategy['ranking_factors'].insert(0, 'price_low')
        elif price_range == 'luxury':
            strategy['ranking_factors'].insert(0, 'price_high')
        
        return strategy
    
    def _search_restaurants(self, coordinates: Dict[str, float], radius: int, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for restaurants using Google Places API"""
        try:
            # Primary search for restaurants
            places = self.gmaps_client.places_nearby(
                location=(coordinates['lat'], coordinates['lng']),
                radius=radius,
                type='restaurant'
            )
            
            restaurants = []
            for place in places.get('results', []):
                restaurant_data = self._extract_restaurant_data(place)
                restaurants.append(restaurant_data)
            
            # If cuisine-specific, do additional searches
            for keyword in strategy['keywords'][:2]:  # Limit to 2 additional searches
                try:
                    keyword_places = self.gmaps_client.places_nearby(
                        location=(coordinates['lat'], coordinates['lng']),
                        radius=radius,
                        keyword=keyword,
                        type='restaurant'
                    )
                    
                    for place in keyword_places.get('results', [])[:5]:  # Top 5 per keyword
                        restaurant_data = self._extract_restaurant_data(place)
                        # Avoid duplicates
                        if not any(r['place_id'] == restaurant_data['place_id'] for r in restaurants):
                            restaurants.append(restaurant_data)
                except Exception as e:
                    print(f"Keyword search error for {keyword}: {e}")
            
            return restaurants
            
        except Exception as e:
            print(f"Restaurant search error: {e}")
            return self._get_mock_restaurants()
    
    def _extract_restaurant_data(self, place: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant restaurant data from Google Places result"""
        return {
            'place_id': place.get('place_id'),
            'name': place.get('name', 'Unknown Restaurant'),
            'rating': place.get('rating', 0),
            'user_ratings_total': place.get('user_ratings_total', 0),
            'price_level': place.get('price_level', 2),
            'address': place.get('vicinity', ''),
            'coordinates': {
                'lat': place['geometry']['location']['lat'],
                'lng': place['geometry']['location']['lng']
            },
            'types': place.get('types', []),
            'opening_hours': place.get('opening_hours', {}),
            'photos': [photo.get('photo_reference') for photo in place.get('photos', [])[:3]]
        }
    
    def _filter_and_rank_restaurants(self, restaurants: List[Dict], query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter and rank restaurants based on user preferences"""
        # Filter by minimum rating
        min_rating = 3.5
        filtered = [r for r in restaurants if r.get('rating', 0) >= min_rating]
        
        # Sort by rating and review count
        filtered.sort(key=lambda x: (x.get('rating', 0) * 0.7 + min(x.get('user_ratings_total', 0) / 100, 5) * 0.3), reverse=True)
        
        return filtered[:10]  # Top 10 restaurants
    
    def _get_mock_restaurants(self) -> List[Dict[str, Any]]:
        """Fallback mock restaurant data"""
        return [
            {
                'place_id': 'mock_restaurant_1',
                'name': 'Sample Restaurant',
                'rating': 4.2,
                'user_ratings_total': 150,
                'price_level': 2,
                'address': 'Sample Address',
                'coordinates': {'lat': 0, 'lng': 0},
                'types': ['restaurant', 'food'],
                'opening_hours': {'open_now': True},
                'photos': []
            }
        ]


class HotelAgent(BaseAgent):
    """Specialized agent for hotel searches with chain-of-thought reasoning"""
    
    def __init__(self, gmaps_key: str = None):
        super().__init__("HotelAgent")
        self.gmaps_key = gmaps_key
        # Initialize Google Maps client with the provided key
        if self.gmaps_key:
            self.gmaps_client = googlemaps.Client(key=self.gmaps_key)
        else:
            self.gmaps_client = None
    
    def execute(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hotel search with chain-of-thought reasoning
        Expected query format: {
            'location': str,
            'coordinates': {'lat': float, 'lng': float},
            'check_in': str,
            'check_out': str,
            'guests': int,
            'hotel_quality': str,
            'amenities_required': List[str],
            'budget_range': str
        }
        """
        self.reasoning_chain = []
        
        # Step 1: Query Analysis
        self.add_reasoning_step(
            step="Hotel Query Analysis",
            reasoning="Analyzing hotel search requirements including dates, guest count, and preferences",
            data={'query': query},
            next_actions=["determine_search_criteria", "fetch_hotels"]
        )
        
        coordinates = query.get('coordinates')
        location = query.get('location')
        check_in = query.get('check_in')
        check_out = query.get('check_out')
        guests = query.get('guests', 2)
        hotel_quality = query.get('hotel_quality', 'medium')
        amenities_required = query.get('amenities_required', [])
        budget_range = query.get('budget_range', 'moderate')
        
        if not coordinates and location:
            coordinates = self._geocode_location(location)
        
        # Step 2: Determine search criteria
        search_criteria = self._determine_search_criteria(hotel_quality, amenities_required, budget_range, guests)
        self.add_reasoning_step(
            step="Search Criteria Determination",
            reasoning="Establishing search parameters based on quality requirements and amenities",
            data=search_criteria,
            next_actions=["execute_hotel_search"]
        )
        
        # Step 3: Execute hotel search
        hotels = self._search_hotels(coordinates, search_criteria)
        self.add_reasoning_step(
            step="Hotel Data Retrieval",
            reasoning="Fetching hotel data from Google Places API based on search criteria",
            data={'found_hotels': len(hotels)},
            next_actions=["evaluate_and_rank_hotels"]
        )
        
        # Step 4: Evaluate and rank hotels
        ranked_hotels = self._evaluate_and_rank_hotels(hotels, query)
        self.add_reasoning_step(
            step="Hotel Evaluation and Ranking",
            reasoning="Evaluating hotels based on ratings, amenities, and user preferences",
            data={'final_hotels': len(ranked_hotels)},
            next_actions=["return_hotel_recommendations"]
        )
        
        return {
            'hotels': ranked_hotels,
            'search_criteria': search_criteria,
            'total_found': len(hotels),
            'reasoning_chain': [
                {
                    'step': r.step,
                    'reasoning': r.reasoning,
                    'next_actions': r.next_actions
                } for r in self.reasoning_chain
            ]
        }
    
    def _geocode_location(self, location: str) -> Dict[str, float]:
        """Geocode location to coordinates"""
        try:
            result = self.gmaps_client.geocode(location)
            if result:
                loc = result[0]['geometry']['location']
                return {'lat': loc['lat'], 'lng': loc['lng']}
        except Exception as e:
            print(f"Geocoding error: {e}")
        return {'lat': 0, 'lng': 0}
    
    def _determine_search_criteria(self, hotel_quality: str, amenities_required: List[str], budget_range: str, guests: int) -> Dict[str, Any]:
        """Determine hotel search criteria"""
        criteria = {
            'types': ['lodging'],
            'min_rating': 3.0,
            'keywords': [],
            'filters': []
        }
        
        # Quality-based criteria
        if hotel_quality == 'luxury':
            criteria['min_rating'] = 4.5
            criteria['keywords'].extend(['luxury', 'resort', '5 star'])
        elif hotel_quality == 'high':
            criteria['min_rating'] = 4.0
            criteria['keywords'].extend(['hotel', '4 star'])
        elif hotel_quality == 'medium':
            criteria['min_rating'] = 3.5
            criteria['keywords'].extend(['hotel'])
        else:  # budget
            criteria['min_rating'] = 3.0
            criteria['keywords'].extend(['budget', 'inn', 'hostel'])
        
        # Amenity-based keywords
        if 'pool' in amenities_required:
            criteria['keywords'].append('pool')
        if 'spa' in amenities_required:
            criteria['keywords'].append('spa')
        if 'gym' in amenities_required:
            criteria['keywords'].append('fitness')
        
        return criteria
    
    def _search_hotels(self, coordinates: Dict[str, float], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for hotels using Google Places API"""
        try:
            # Primary search for lodging
            places = self.gmaps_client.places_nearby(
                location=(coordinates['lat'], coordinates['lng']),
                radius=5000,  # 5km radius
                type='lodging'
            )
            
            hotels = []
            for place in places.get('results', []):
                hotel_data = self._extract_hotel_data(place)
                hotels.append(hotel_data)
            
            return hotels
            
        except Exception as e:
            print(f"Hotel search error: {e}")
            return self._get_mock_hotels()
    
    def _extract_hotel_data(self, place: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant hotel data from Google Places result"""
        return {
            'place_id': place.get('place_id'),
            'name': place.get('name', 'Unknown Hotel'),
            'rating': place.get('rating', 0),
            'user_ratings_total': place.get('user_ratings_total', 0),
            'price_level': place.get('price_level', 2),
            'address': place.get('vicinity', ''),
            'coordinates': {
                'lat': place['geometry']['location']['lat'],
                'lng': place['geometry']['location']['lng']
            },
            'types': place.get('types', []),
            'photos': [photo.get('photo_reference') for photo in place.get('photos', [])[:3]]
        }
    
    def _evaluate_and_rank_hotels(self, hotels: List[Dict], query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate and rank hotels based on criteria"""
        hotel_quality = query.get('hotel_quality', 'medium')
        
        # Filter by minimum rating based on quality
        min_ratings = {'luxury': 4.5, 'high': 4.0, 'medium': 3.5, 'budget': 3.0}
        min_rating = min_ratings.get(hotel_quality, 3.5)
        
        filtered = [h for h in hotels if h.get('rating', 0) >= min_rating]
        
        # Sort by rating and review count
        filtered.sort(key=lambda x: (x.get('rating', 0) * 0.8 + min(x.get('user_ratings_total', 0) / 200, 5) * 0.2), reverse=True)
        
        return filtered[:8]  # Top 8 hotels
    
    def _get_mock_hotels(self) -> List[Dict[str, Any]]:
        """Fallback mock hotel data"""
        return [
            {
                'place_id': 'mock_hotel_1',
                'name': 'Sample Hotel',
                'rating': 4.1,
                'user_ratings_total': 200,
                'price_level': 3,
                'address': 'Sample Hotel Address',
                'coordinates': {'lat': 0, 'lng': 0},
                'types': ['lodging'],
                'photos': []
            }
        ]


# =============================================================================
# MAIN ORCHESTRATOR WITH CHAIN-OF-THOUGHT REASONING
# =============================================================================
class TravelPlannerOrchestrator:
    """Main orchestrator that coordinates all specialized agents using chain-of-thought reasoning"""
    
    def __init__(self, gmaps_key: str = None, weather_key: str = None, gemini_key: str = None):
        """
        Initialize the orchestrator with API keys
        
        Args:
            gmaps_key: Google Maps API key
            weather_key: OpenWeatherMap API key  
            gemini_key: Google AI (Gemini) API key
        """
        # Store API keys
        self.gmaps_key = gmaps_key
        self.weather_key = weather_key
        self.gemini_key = gemini_key
        
        # Initialize Gemini client
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.gemini_model = None
            
        # Initialize Google Maps client for orchestrator-level operations
        if self.gmaps_key:
            self.gmaps_client = googlemaps.Client(key=self.gmaps_key)
        else:
            self.gmaps_client = None
        
        # Initialize agents with API keys
        self.weather_agent = WeatherAgent(weather_key=self.weather_key)
        self.restaurant_agent = RestaurantAgent(gmaps_key=self.gmaps_key)
        self.hotel_agent = HotelAgent(gmaps_key=self.gmaps_key)
        self.reasoning_chain = []
    
    def _normalize_user_prefs(self, user_prefs: UserPreferences) -> Dict[str, Any]:
        """Normalize user preferences to handle both old and new attribute names"""
        # Convert to dict and handle attribute name mapping
        if hasattr(user_prefs, 'dict'):
            # Pydantic model
            prefs_dict = user_prefs.dict()
        else:
            # Dataclass or dict
            prefs_dict = user_prefs.__dict__ if hasattr(user_prefs, '__dict__') else user_prefs
        
        # Create normalized structure
        normalized = {
            'location': prefs_dict.get('city', prefs_dict.get('location')),
            'numberOfAdults': prefs_dict.get('number_of_adults', prefs_dict.get('numberOfAdults')),
            'numberOfChildren': prefs_dict.get('number_of_children', prefs_dict.get('numberOfChildren')),
            'ticketsBooked': prefs_dict.get('main_tickets_booked', prefs_dict.get('ticketsBooked')),
            'arrivalDateTime': prefs_dict.get('start_datetime', prefs_dict.get('arrivalDateTime')),
            'departureDateTime': prefs_dict.get('end_datetime', prefs_dict.get('departureDateTime')),
            'localTransport': prefs_dict.get('local_transport', prefs_dict.get('localTransport')),
            'hotelQuality': prefs_dict.get('hotel_quality', prefs_dict.get('hotelQuality')),
            'poolRequired': prefs_dict.get('pool_required', prefs_dict.get('poolRequired')),
            'budget': prefs_dict.get('budget_per_person', prefs_dict.get('budget')),
            'interests': prefs_dict.get('interests', []),
            'dietary_restrictions': prefs_dict.get('dietary_restrictions', [])
        }
        
        return normalized
    
    def create_enhanced_itinerary(self, preferences: UserPreferences, swipe_decisions: List[SwipeDecision], all_locations: List[Dict]) -> Dict[str, Any]:
        """Create itinerary using chain-of-thought reasoning and specialized agents"""
        self.reasoning_chain = []
        
        # Normalize user preferences to handle attribute name changes
        normalized_prefs = self._normalize_user_prefs(preferences)
        
        # Step 1: Master Planning Analysis
        self._add_reasoning_step(
            "Master Planning Analysis",
            "Analyzing user preferences, travel dates, and location choices to create a comprehensive travel strategy",
            {'user_prefs': normalized_prefs, 'swipe_decisions': len(swipe_decisions)},
            ["coordinate_specialized_agents", "synthesize_recommendations"]
        )
        
        # Extract and process user data
        coordinates = self._geocode_location(normalized_prefs['location'])
        
        # Handle date extraction safely
        arrival_dt = normalized_prefs.get('arrivalDateTime') or normalized_prefs.get('start_datetime')
        departure_dt = normalized_prefs.get('departureDateTime') or normalized_prefs.get('end_datetime')
        
        if arrival_dt and departure_dt:
            start_date = arrival_dt[:10] if len(arrival_dt) >= 10 else arrival_dt
            end_date = departure_dt[:10] if len(departure_dt) >= 10 else departure_dt
        else:
            # Default dates if not provided
            today = datetime.now()
            start_date = today.strftime('%Y-%m-%d')
            end_date = (today + timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Process swipe decisions
        super_liked, liked = self._process_swipe_decisions(swipe_decisions, all_locations)
        
        # Step 2: Coordinate specialized agents
        self._add_reasoning_step(
            "Agent Coordination",
            "Coordinating weather, restaurant, and hotel agents to gather comprehensive travel data",
            {'agents_to_coordinate': 3},
            ["execute_weather_analysis", "execute_restaurant_search", "execute_hotel_search"]
        )
        
        # Execute specialized agents in parallel (conceptually)
        weather_data = self.weather_agent.execute({
            'location': normalized_prefs['location'],
            'coordinates': coordinates,
            'dates': [start_date, end_date]
        })
        
        restaurant_data = self.restaurant_agent.execute({
            'location': normalized_prefs['location'],
            'coordinates': coordinates,
            'cuisine_preferences': normalized_prefs['interests'],
            'dietary_restrictions': normalized_prefs['dietary_restrictions'],
            'price_range': self._budget_to_price_range(normalized_prefs['budget'])
        })
        
        hotel_data = self.hotel_agent.execute({
            'location': normalized_prefs['location'],
            'coordinates': coordinates,
            'check_in': start_date,
            'check_out': end_date,
            'guests': normalized_prefs['numberOfAdults'] + normalized_prefs['numberOfChildren'],
            'hotel_quality': normalized_prefs['hotelQuality'],
            'amenities_required': ['pool'] if normalized_prefs['poolRequired'] else []
        })
        
        # Step 3: Synthesize all data
        self._add_reasoning_step(
            "Data Synthesis",
            "Combining weather, dining, and accommodation data with user preferences to create optimal itinerary",
            {
                'weather_insights': len(weather_data.get('reasoning_chain', [])),
                'restaurants_found': len(restaurant_data.get('restaurants', [])),
                'hotels_found': len(hotel_data.get('hotels', []))
            },
            ["generate_day_by_day_itinerary"]
        )
        
        # Step 4: Generate comprehensive itinerary
        itinerary = self._generate_comprehensive_itinerary(
            normalized_prefs, super_liked, liked, weather_data, restaurant_data, hotel_data
        )
        
        self._add_reasoning_step(
            "Itinerary Generation",
            "Creating final day-by-day itinerary with optimal scheduling and recommendations",
            {'itinerary_days': len(itinerary.get('daily_schedule', []))},
            ["return_complete_travel_plan"]
        )
        
        return {
            'itinerary': itinerary,
            'weather_insights': weather_data,
            'restaurant_recommendations': restaurant_data,
            'hotel_recommendations': hotel_data,
            'master_reasoning_chain': [
                {
                    'step': r['step'],
                    'reasoning': r['reasoning'],
                    'next_actions': r['next_actions']
                } for r in self.reasoning_chain
            ]
        }
    
    def _add_reasoning_step(self, step: str, reasoning: str, data: Dict, next_actions: List[str]):
        """Add reasoning step to master chain"""
        self.reasoning_chain.append({
            'step': step,
            'reasoning': reasoning,
            'data_gathered': data,
            'next_actions': next_actions
        })
    
    def _geocode_location(self, location: str) -> Dict[str, float]:
        """Geocode location to coordinates"""
        try:
            if self.gmaps_client:
                result = self.gmaps_client.geocode(location)
                if result:
                    loc = result[0]['geometry']['location']
                    return {'lat': loc['lat'], 'lng': loc['lng']}
        except Exception as e:
            print(f"Geocoding error: {e}")
        return {'lat': 0, 'lng': 0}
    
    def _process_swipe_decisions(self, decisions: List[SwipeDecision], locations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Process swipe decisions into super liked and liked locations"""
        location_map = {loc['id']: loc for loc in locations}
        
        super_liked = [location_map[d.locationId] for d in decisions if d.choice == 'superlike' and d.locationId in location_map]
        liked = [location_map[d.locationId] for d in decisions if d.choice == 'right' and d.locationId in location_map]
        
        return super_liked, liked
    
    def _budget_to_price_range(self, budget: int) -> str:
        """Convert budget to price range category"""
        if budget < 30000:
            return 'budget'
        elif budget < 60000:
            return 'moderate'
        elif budget < 100000:
            return 'high'
        else:
            return 'luxury'
    
    def _generate_comprehensive_itinerary(self, user_prefs: Dict[str, Any], super_liked: List[Dict], 
                                        liked: List[Dict], weather_data: Dict, restaurant_data: Dict, 
                                        hotel_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive itinerary using chain-of-thought reasoning"""
        
        # Create chain-of-thought prompt for Gemini
        cot_prompt = self._create_chain_of_thought_prompt(
            user_prefs, super_liked, liked, weather_data, restaurant_data, hotel_data
        )
        
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(cot_prompt)
                return self._parse_gemini_response(response.text)
            else:
                print("Gemini model not available, using fallback itinerary generation")
                return self._generate_fallback_itinerary(user_prefs, super_liked, liked)
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_itinerary(user_prefs, super_liked, liked)
    
    def _create_chain_of_thought_prompt(self, user_prefs: Dict[str, Any], super_liked: List[Dict], 
                                      liked: List[Dict], weather_data: Dict, restaurant_data: Dict, 
                                      hotel_data: Dict) -> str:
        """Create a comprehensive chain-of-thought prompt for Gemini"""
        
        return f"""
You are an expert travel planner creating a comprehensive itinerary. Use chain-of-thought reasoning to create the best possible travel plan.

STEP 1: ANALYZE THE TRAVEL CONTEXT
Location: {user_prefs['location']}
Travel Dates: {user_prefs.get('arrivalDateTime', user_prefs.get('start_datetime', 'N/A'))} to {user_prefs.get('departureDateTime', user_prefs.get('end_datetime', 'N/A'))}
Travelers: {user_prefs['numberOfAdults']} adults, {user_prefs['numberOfChildren']} children
Budget: {user_prefs['budget']} INR
Interests: {user_prefs['interests']}
Transport: {user_prefs['localTransport']}

STEP 2: WEATHER ANALYSIS
Current Weather: {weather_data.get('current_weather', {})}
Weather Advice: {weather_data.get('travel_advice', {})}
Reasoning: Consider how weather impacts daily activities and clothing recommendations.

STEP 3: ACCOMMODATION ANALYSIS
Top Hotels Found: {len(hotel_data.get('hotels', []))}
Hotel Quality Preference: {user_prefs['hotelQuality']}
Pool Required: {user_prefs['poolRequired']}
Reasoning: Match hotel recommendations to user quality expectations and amenity requirements.

STEP 4: DINING ANALYSIS
Restaurants Found: {len(restaurant_data.get('restaurants', []))}
Top Restaurant: {restaurant_data.get('restaurants', [{}])[0].get('name', 'N/A') if restaurant_data.get('restaurants') else 'N/A'}
Reasoning: Select restaurants that match interests and provide variety across the trip.

STEP 5: PREFERRED LOCATIONS INTEGRATION
Super Liked Locations: {[loc['name'] for loc in super_liked]}
Liked Locations: {[loc['name'] for loc in liked]}
Reasoning: Prioritize super-liked locations and integrate liked locations where schedule permits.

STEP 6: CREATE DAY-BY-DAY ITINERARY
Now create a detailed day-by-day itinerary that:
1. Optimizes travel time between locations
2. Considers weather conditions for outdoor/indoor activities
3. Includes meal recommendations at appropriate times
4. Balances super-liked and liked locations
5. Stays within budget constraints
6. Matches the local transport method

Please respond with ONLY valid JSON in this exact format:
{{
  "trip_overview": {{
    "destination": "{user_prefs['location']}",
    "duration_days": 3,
    "total_budget_estimate": 75000,
    "weather_summary": "Pleasant with occasional rain",
    "travel_style": "Adventure and Culture"
  }},
  "daily_schedule": [
    {{
      "day": 1,
      "date": "2025-12-20",
      "theme": "Arrival and Beach Relaxation",
      "activities": [
        {{
          "time": "09:00",
          "activity": "Check-in at Hotel",
          "location": "Hotel Name",
          "duration": "1 hour",
          "cost_estimate": 0,
          "weather_consideration": "Indoor activity"
        }},
        {{
          "time": "11:00",
          "activity": "Beach Visit",
          "location": "Baga Beach",
          "duration": "3 hours",
          "cost_estimate": 500,
          "weather_consideration": "Check UV index"
        }}
      ],
      "meals": [
        {{
          "meal": "lunch",
          "time": "13:00",
          "restaurant": "Restaurant Name",
          "cuisine": "Goan",
          "cost_estimate": 1200
        }}
      ],
      "accommodation": "Hotel Name",
      "daily_budget": 25000
    }}
  ],
  "recommended_hotels": [
    {{
      "name": "Top Hotel Choice",
      "rating": 4.2,
      "price_per_night": 8000,
      "amenities": ["Pool", "WiFi"],
      "reasoning": "Matches quality preference and has required pool"
    }}
  ],
  "recommended_restaurants": [
    {{
      "name": "Top Restaurant Choice",
      "cuisine": "Local",
      "rating": 4.5,
      "price_range": "Moderate",
      "reasoning": "Highly rated local cuisine experience"
    }}
  ],
  "packing_suggestions": [
    "Light cotton clothes",
    "Sunscreen",
    "Umbrella"
  ],
  "transportation_plan": {{
    "local_transport": "{user_prefs['localTransport']}",
    "daily_transport_budget": 800,
    "transport_tips": "Book scooter rental in advance"
  }},
  "budget_breakdown": {{
    "accommodation": 24000,
    "food": 18000,
    "activities": 15000,
    "transport": 8000,
    "miscellaneous": 5000,
    "total": 70000
  }}
}}
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response with improved error handling"""
        try:
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return self._generate_fallback_itinerary_dict()
    
    def _generate_fallback_itinerary(self, user_prefs: Dict[str, Any], super_liked: List[Dict], liked: List[Dict]) -> Dict[str, Any]:
        """Generate fallback itinerary when Gemini fails"""
        return self._generate_fallback_itinerary_dict()
    
    def _generate_fallback_itinerary_dict(self) -> Dict[str, Any]:
        """Generate a basic fallback itinerary structure"""
        return {
            "trip_overview": {
                "destination": "Travel Destination",
                "duration_days": 2,
                "total_budget_estimate": 20000,
                "weather_summary": "Pleasant weather expected",
                "travel_style": "Mixed activities"
            },
            "daily_schedule": [
                {
                    "day": 1,
                    "date": "2025-12-20",
                    "theme": "Arrival and Exploration",
                    "activities": [
                        {
                            "time": "10:00",
                            "activity": "Check-in and local exploration",
                            "location": "City center",
                            "duration": "2 hours",
                            "cost_estimate": 1000,
                            "weather_consideration": "Flexible indoor/outdoor"
                        }
                    ],
                    "meals": [
                        {
                            "meal": "lunch",
                            "time": "13:00",
                            "restaurant": "Local Restaurant",
                            "cuisine": "Local",
                            "cost_estimate": 1500
                        }
                    ],
                    "accommodation": "Recommended Hotel",
                    "daily_budget": 25000
                }
            ],
            "recommended_hotels": [
                {
                    "name": "Sample Hotel",
                    "rating": 4.0,
                    "price_per_night": 8000,
                    "amenities": ["WiFi", "AC"],
                    "reasoning": "Good value for money"
                }
            ],
            "packing_suggestions": [
                "Comfortable clothing",
                "Travel essentials",
                "Weather appropriate gear"
            ]
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main function to demonstrate the enhanced travel planner"""
    
    # Initialize the orchestrator
    orchestrator = TravelPlannerOrchestrator()
    
    # Sample user preferences
    user_prefs = UserPreferences(
        city="Goa, India",
        city_id="1",
        main_tickets_booked=False,
        number_of_adults=2,
        number_of_children=0,
        budget_per_person=80000,
        local_transport="scooter",
        hotel_quality="medium",
        pool_required=True,
        interests=["beach", "adventure", "food", "history"],
        start_datetime="2025-12-20T08:00:00Z",
        end_datetime="2025-12-22T20:00:00Z",
        dietary_restrictions=["vegetarian"]
    )
    
    # Sample swipe decisions
    decisions = [
        SwipeDecision(locationId="1", choice="superlike"),
        SwipeDecision(locationId="4", choice="superlike"),
        SwipeDecision(locationId="2", choice="right"),
        SwipeDecision(locationId="6", choice="right")
    ]
    
    # Sample locations
    goa_locations = [
        {"id": "1", "name": "Baga Beach", "description": "Popular beach with water sports", "tags": ["beach", "nightlife"], "city": "Goa", "area": "North Goa", "image": "", "coordinates": {"lat": 15.5583, "lng": 73.7517}, "city_id": 1},
        {"id": "2", "name": "Calangute Beach", "description": "Crowded but beautiful beach", "tags": ["beach", "food"], "city": "Goa", "area": "North Goa", "image": "", "coordinates": {"lat": 15.5478, "lng": 73.7592}, "city_id": 1},
        {"id": "4", "name": "Dudhsagar Falls", "description": "Spectacular waterfall", "tags": ["nature", "adventure"], "city": "Goa", "area": "South Goa", "image": "", "coordinates": {"lat": 15.3144, "lng": 74.3145}, "city_id": 1},
        {"id": "6", "name": "Basilica of Bom Jesus", "description": "Historic church", "tags": ["history", "religion"], "city": "Goa", "area": "Old Goa", "image": "", "coordinates": {"lat": 15.5009, "lng": 73.9115}, "city_id": 1}
    ]
    
    # Generate enhanced itinerary
    print("Generating enhanced travel itinerary with chain-of-thought reasoning...")
    print("=" * 80)
    
    result = orchestrator.create_enhanced_itinerary(user_prefs, decisions, goa_locations)
    
    # Display results
    print("\n🧠 MASTER REASONING CHAIN:")
    print("-" * 40)
    for i, step in enumerate(result['master_reasoning_chain'], 1):
        print(f"{i}. {step['step']}")
        print(f"   Reasoning: {step['reasoning']}")
        print(f"   Next Actions: {', '.join(step['next_actions'])}")
        print()
    
    print("\n🌤️  WEATHER INSIGHTS:")
    print("-" * 40)
    weather_chain = result['weather_insights'].get('reasoning_chain', [])
    for step in weather_chain:
        print(f"• {step['step']}: {step['reasoning']}")
    
    print(f"\nCurrent Weather: {result['weather_insights']['current_weather']['description']}, {result['weather_insights']['current_weather']['temperature']}°C")
    print(f"Travel Advice: {result['weather_insights']['travel_advice']['clothing_recommendations']}")
    
    print("\n🍽️  RESTAURANT RECOMMENDATIONS:")
    print("-" * 40)
    restaurants = result['restaurant_recommendations']['restaurants'][:3]
    for restaurant in restaurants:
        print(f"• {restaurant['name']} - Rating: {restaurant['rating']}/5 - {restaurant['address']}")
    
    print("\n🏨 HOTEL RECOMMENDATIONS:")
    print("-" * 40)
    hotels = result['hotel_recommendations']['hotels'][:3]
    for hotel in hotels:
        print(f"• {hotel['name']} - Rating: {hotel['rating']}/5 - {hotel['address']}")
    
    print("\n📅 FINAL ITINERARY:")
    print("-" * 40)
    print(json.dumps(result['itinerary'], indent=2))


if __name__ == '__main__':
    main()