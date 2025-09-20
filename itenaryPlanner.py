import requests
import json
import logging
from gemini_cot import TravelPlannerOrchestrator
from gemini_v0 import GeminiV0TravelPlanner


logger = logging.getLogger(__name__)


class ItineraryPlanner:
    """
    Generates a personalized, optimized travel itinerary by integrating with external APIs.
    Uses the enhanced TravelPlannerOrchestrator for comprehensive itinerary generation.
    """
    def __init__(self, gmaps_key: str = None, weather_key: str = None, gemini_key: str = None):
        """
        Initializes the planner with necessary API keys and loads all location data.

        Args:
            gmaps_key: Your Google Maps Platform API key (optional).
            weather_key: Your OpenWeatherMap API key (optional).
            gemini_key: Your Google AI (Gemini) API key (optional).
        """
        # Store keys
        self.gmaps_key = gmaps_key
        self.weather_key = weather_key
        self.gemini_key = gemini_key
        self.session = requests.Session() # Use a session for connection pooling
        
        # Initialize the enhanced travel planner orchestrator
        self.enhanced_planner = TravelPlannerOrchestrator(
            gmaps_key=gmaps_key,
            weather_key=weather_key,
            gemini_key=gemini_key
        )
        
        # Initialize the Gemini v0 travel planner
        self.gemini_v0_planner = GeminiV0TravelPlanner(
            gmaps_key=gmaps_key,
            weather_key=weather_key,
            gemini_key=gemini_key
        )
        
        # Initialize location storage
        self.all_locations = []
        self.locations_by_city = {}
        
        # Load all locations at startup
        try:
            file_path = "database/prominent_locations.json"
            with open(file_path, 'r', encoding='utf-8') as file:
                self.all_locations = json.load(file)
            logger.info(f"Successfully loaded {len(self.all_locations)} locations on startup from {file_path}")
            
            # Pre-populate locations_by_city cache
            for location in self.all_locations:
                city_id_key = str(location.get("city_id", ""))
                if city_id_key:
                    if city_id_key not in self.locations_by_city:
                        self.locations_by_city[city_id_key] = []
                    self.locations_by_city[city_id_key].append(location)
            
            logger.info(f"Cached locations for {len(self.locations_by_city)} cities")
        except FileNotFoundError:
            logger.error(f"File 'database/prominent_locations.json' not found during initialization")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in 'database/prominent_locations.json' during initialization")
        except Exception as e:
            logger.error(f"Error loading locations during initialization: {str(e)}")

    def test_itinerary(self):
        """
        Returns the provided itinerary data directly. Useful for testing and development.
        
        Returns:
            The provided itinerary data
        """
        itinerary_data = {"itenary"}
        if isinstance(itinerary_data, str):
            try:
                # If it's a string, parse it as JSON
                return json.loads(itinerary_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided")
        
        # If it's already a dict or another data structure, return as is
        return itinerary_data
    
    def get_prominent_locations(self, preferences):
        """
        Returns a list of prominent locations filtered by city ID using preloaded data.
        
        Args:
            preferences: User's travel preferences with city_id
            
        Returns:
            list: A list of prominent locations matching the city ID
        """
        city_id = preferences.city_id if hasattr(preferences, 'city_id') else None
        logger.info(f"Finding prominent locations for city_id: {city_id}")
        
        if not city_id:
            logger.warning("No city_id provided in preferences")
            return []
        
        # Use preloaded cached locations for this city_id
        city_id_key = str(city_id)  # Convert to string for consistent dictionary keys
        
        if city_id_key in self.locations_by_city:
            logger.info(f"Using cached locations for city_id: {city_id}")
            filtered_locations = self.locations_by_city[city_id_key]
        else:
            logger.warning(f"No locations found for city_id: {city_id}")
            filtered_locations = []
            
        # Apply interest filtering if requested
        if hasattr(preferences, 'interests') and preferences.interests:
            interest_filtered = []
            for location in filtered_locations:
                # Check if any tag matches any interest
                if any(interest.lower() in [tag.lower() for tag in location.get("tags", [])] 
                    for interest in preferences.interests):
                    interest_filtered.append(location)
            
            if interest_filtered:  # Only use interest filtering if we got results
                result_locations = interest_filtered
                logger.info(f"Applied interest filtering, found {len(result_locations)} matching locations")
            else:
                result_locations = filtered_locations
        else:
            result_locations = filtered_locations
        
        logger.info(f"Returning {len(result_locations)} locations for city_id: {city_id}")
        
        # Transform coordinates from lat/lng to latitude/longitude for frontend compatibility
        transformed_locations = []
        for location in result_locations:
            transformed_location = location.copy()
            if 'coordinates' in transformed_location:
                coords = transformed_location['coordinates']
                if 'lat' in coords and 'lng' in coords:
                    transformed_location['coordinates'] = {
                        'latitude': coords['lat'],
                        'longitude': coords['lng']
                    }
            transformed_locations.append(transformed_location)
        
        return transformed_locations
    

    def get_all_locations(self):
        """
        Returns all locations that have been loaded with transformed coordinates.
        
        Returns:
            list: All loaded locations with latitude/longitude coordinates
        """
        if hasattr(self, 'all_locations'):
            # Transform coordinates from lat/lng to latitude/longitude for frontend compatibility
            transformed_locations = []
            for location in self.all_locations:
                transformed_location = location.copy()
                if 'coordinates' in transformed_location:
                    coords = transformed_location['coordinates']
                    if 'lat' in coords and 'lng' in coords:
                        transformed_location['coordinates'] = {
                            'latitude': coords['lat'],
                            'longitude': coords['lng']
                        }
                transformed_locations.append(transformed_location)
            return transformed_locations
        else:
            logger.warning("No locations have been loaded yet")
            return []

    def get_cached_locations_by_city(self, city_id):
        """
        Returns cached locations for a specific city_id.
        
        Args:
            city_id: The city ID to retrieve locations for
            
        Returns:
            list: Locations for the specified city_id
        """
        if hasattr(self, 'locations_by_city') and city_id in self.locations_by_city:
            return self.locations_by_city[city_id]
        else:
            logger.warning(f"No cached locations for city_id: {city_id}")
            return []
        
    def generate_itenary(self, preferences, swipe_decisions):
        """
        Generates a personalized travel itinerary using the enhanced TravelPlannerOrchestrator.
        
        Args:
            preferences: User's travel preferences (supports both old and new formats)
            swipe_decisions: User's decisions on locations (like, superlike, or dislike)
            
        Returns:
            Dict containing comprehensive itinerary with reasoning chains
        """
        logger.info(f"Generating enhanced itinerary for {getattr(preferences, 'city', getattr(preferences, 'location', 'unknown destination'))}")
        logger.info(f"User preferences received: {type(preferences)}")
        logger.info(f"Swipe decisions count: {len(swipe_decisions) if swipe_decisions else 0}")
        
        # Use the enhanced planner to create comprehensive itinerary
        try:
            result = self.enhanced_planner.create_enhanced_itinerary(
                preferences=preferences,
                swipe_decisions=swipe_decisions,
                all_locations=self.all_locations
            )
            
            logger.info("✅ Enhanced itinerary generation completed successfully")
            logger.info(f"✅ Generated itinerary with {len(result.get('master_reasoning_chain', []))} reasoning steps")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced itinerary generation: {e}")
            # Fallback to simple response
            return {
                'itinerary': {
                    'trip_overview': {
                        'destination': getattr(preferences, 'city', getattr(preferences, 'location', 'Unknown')),
                        'duration_days': 2,
                        'total_budget_estimate': 20000,
                        'weather_summary': 'Weather data unavailable',
                        'travel_style': 'Basic planning'
                    },
                    'daily_schedule': [],
                    'recommended_hotels': [],
                    'packing_suggestions': ['Basic travel essentials']
                },
                'weather_insights': {'current_weather': {}, 'travel_advice': {}},
                'restaurant_recommendations': {'restaurants': []},
                'hotel_recommendations': {'hotels': []},
                'master_reasoning_chain': [
                    {
                        'step': 'Error Handling',
                        'reasoning': f'Fallback mode activated due to error: {str(e)}',
                        'next_actions': ['return_basic_itinerary']
                    }
                ],
                'error': str(e)
            }
    
    def generate_itenaryv0(self, preferences, swipe_decisions):
        """
        Generates a travel itinerary using the Gemini v0 method.
        This is an alternative approach that uses direct Gemini AI generation
        with structured JSON output.
        
        Args:
            preferences: User's travel preferences (new UserPreferences format)
            swipe_decisions: User's decisions on locations (like, superlike, or dislike)
            
        Returns:
            Dict containing itinerary generated by Gemini AI
        """
        logger.info(f"Generating v0 itinerary for {getattr(preferences, 'city', 'unknown destination')}")
        logger.info(f"User preferences received: {type(preferences)}")
        logger.info(f"Swipe decisions count: {len(swipe_decisions) if swipe_decisions else 0}")
        
        try:
            # Convert locations to the format expected by gemini_v0
            # We need to convert our location dict format to ProminentLocation objects
            from models import ProminentLocation, Coordinates
            
            prominent_locations = []
            for location_dict in self.all_locations:
                # Create Coordinates object
                coords = Coordinates(
                    latitude=location_dict.get('coordinates', {}).get('latitude', 0.0),
                    longitude=location_dict.get('coordinates', {}).get('longitude', 0.0)
                )
                
                # Create ProminentLocation object
                location = ProminentLocation(
                    id=str(location_dict.get('id', '')),
                    name=location_dict.get('name', ''),
                    description=location_dict.get('description', ''),
                    tags=location_dict.get('tags', []),
                    city=location_dict.get('city', ''),
                    city_id=location_dict.get('city_id', 0),
                    area=location_dict.get('area', ''),
                    image=location_dict.get('image', ''),
                    coordinates=coords
                )
                prominent_locations.append(location)
            
            # Call the gemini_v0 generation function
            result = self.gemini_v0_planner.generate_itinerary(
                preferences=preferences,
                decisions=swipe_decisions,
                locations=prominent_locations
            )
            
            logger.info("✅ Gemini v0 itinerary generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in Gemini v0 itinerary generation: {e}")
            # Fallback to simple response
            return {
                'success': False,
                'error': str(e),
                'itinerary': {
                    'tripTitle': f"Trip to {getattr(preferences, 'city', 'Unknown')}",
                    'days': [
                        {
                            'date': '2024-01-01',
                            'day': 1,
                            'theme': 'Arrival and Exploration',
                            'events': [
                                {
                                    'time': '10:00 AM',
                                    'title': 'Arrival',
                                    'description': 'Arrive at destination',
                                    'type': 'travel'
                                }
                            ]
                        }
                    ]
                },
                'method': 'gemini_v0_fallback'
            }
       