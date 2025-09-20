import os
import json
import google.generativeai as genai
from fastapi import HTTPException
from typing import List, Dict, Any
from datetime import datetime

# Import the data models from the shared models module
from models import UserPreferences, SwipeDecision, ProminentLocation


class GeminiV0TravelPlanner:
    """
    Travel planner using Gemini AI with direct structured JSON generation.
    This is an alternative approach to the enhanced chain-of-thought planner.
    """
    
    def __init__(self, gmaps_key: str = None, weather_key: str = None, gemini_key: str = None):
        """
        Initialize the Gemini v0 planner with API keys
        
        Args:
            gmaps_key: Google Maps API key (stored but not used in v0)
            weather_key: OpenWeatherMap API key (stored but not used in v0)
            gemini_key: Google AI (Gemini) API key
        """
        # Store API keys
        self.gmaps_key = gmaps_key
        self.weather_key = weather_key
        self.gemini_key = gemini_key
        
        # Initialize Gemini client
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.gemini_model = None
            print("Warning: Gemini API key not provided")
    
    def get_location_name_by_id(self, location_id: str, locations: List[ProminentLocation]) -> str:
        """Get location name by ID from the locations list"""
        for location in locations:
            if location.id == location_id:
                return location.name
        return f"Location ID: {location_id}"

    def create_gemini_prompt(self, preferences: UserPreferences, decisions: List[SwipeDecision], locations: List[ProminentLocation]) -> str:
        """Create the prompt for Gemini AI based on the new UserPreferences structure"""
        
        # Extract liked locations based on choice field
        super_liked = [self.get_location_name_by_id(d.locationId, locations) 
                       for d in decisions if d.choice == 'superlike']
        liked = [self.get_location_name_by_id(d.locationId, locations) 
                 for d in decisions if d.choice == 'right']
        
        travel_prompt_section = ''
        task_structure_prompt = ''
        
        # Handle main tickets booked scenario using new field names
        if preferences.main_tickets_booked and preferences.start_datetime and preferences.end_datetime:
            try:
                # Handle different datetime formats
                arrival_str = preferences.start_datetime
                departure_str = preferences.end_datetime
                
                # Remove Z and handle timezone if needed
                if arrival_str.endswith('Z'):
                    arrival_str = arrival_str[:-1] + '+00:00'
                if departure_str.endswith('Z'):
                    departure_str = departure_str[:-1] + '+00:00'
                    
                arrival = datetime.fromisoformat(arrival_str)
                departure = datetime.fromisoformat(departure_str)
                duration_ms = (departure - arrival).total_seconds() * 1000
                duration_days = max(1, int((duration_ms / (1000 * 60 * 60 * 24)) + 0.5))
                
                travel_prompt_section = f"""
                - **Travel Plan:** Main tickets are already booked.
                - **Arrival:** The user will arrive at {preferences.start_location or 'destination'} on {arrival.strftime('%A, %B %d, %Y at %I:%M %p')}.
                - **Departure:** The user will depart from {preferences.end_location or 'destination'} on {departure.strftime('%A, %B %d, %Y at %I:%M %p')}.
                - **Trip Duration:** The total duration is {duration_days} days.
                """
                task_structure_prompt = f"Create a plan for each of the {duration_days} days. Use the provided arrival date to generate realistic dates for the itinerary."
            except Exception as e:
                print(f"Error parsing booked ticket dates: {e}")
                task_structure_prompt = "Create a 3-day itinerary as a default."
        else:
            # Handle flexible travel scenario (tickets not booked)
            travel_prompt_section = f"""
            - **Travel Plan:** Main tickets are NOT booked. The itinerary should include suggestions for travel arrangements.
            - **Travel Flexibility:** The user has flexibility in their travel dates.
            """
            task_structure_prompt = "Create a 3-day itinerary as a default, including travel arrangement suggestions."
        
        # Handle dietary restrictions
        dietary_info = ""
        if preferences.dietary_restrictions:
            dietary_info = f"- **Dietary Restrictions:** {', '.join(preferences.dietary_restrictions)}"
        
        prompt = f"""
        You are an expert travel agent creating a personalized itinerary for a trip to {preferences.city}.

        **User Preferences:**
        - **Destination:** {preferences.city}
        - **Travelers:** {preferences.number_of_adults} adults and {preferences.number_of_children} children.
        {travel_prompt_section}
        - **Local Transport:** The user prefers to use {preferences.local_transport}.
        - **Accommodation:** A {preferences.hotel_quality} quality hotel. A pool is {'required' if preferences.pool_required else 'not required'}.
        - **Budget:** Approximately INR {preferences.budget_per_person} per person for the entire trip.
        - **Primary Interests:** {', '.join(preferences.interests) if preferences.interests else 'General sightseeing'}.
        {dietary_info}
        - **User's Trip Description:** "{preferences.description_of_trip or 'No specific description provided.'}"

        **User's Location Choices:**
        - **Must-Visit Locations (Super Liked):** {', '.join(super_liked) if super_liked else 'None specified.'}
        - **Interested Locations (Liked):** {', '.join(liked) if liked else 'None specified.'}

        **Your Task:**
        Create a detailed, optimized, day-by-day itinerary.
        1. **Structure:** {task_structure_prompt}
        2. **Incorporate Choices:** You MUST include all "Must-Visit" locations. You should try to include as many "Interested Locations" as logically possible.
        3. **Optimization:** Group activities geographically to minimize travel time. Suggest logical travel flow.
        4. **Completeness:** Add suggestions for meals (breakfast, lunch, dinner) and travel between locations that fit the user's budget and preferences. If children are present, suggest family-friendly options.
        5. **Be Realistic:** Ensure the schedule is not too packed. Allow for free time and relaxation.

        Please provide the output in a valid JSON format that adheres to the provided schema.
        """
        
        return prompt

    def generate_itinerary(self, preferences: UserPreferences, decisions: List[SwipeDecision], locations: List[ProminentLocation]) -> Dict[str, Any]:
        """
        Generate an itinerary using Gemini AI based on user preferences, swipe decisions, and available locations
        """
        try:
            print("=== DEBUG: Received request for v0 itinerary generation ===")
            print(f"User Preferences: {preferences}")
            print(f"Swipe Decisions: {decisions}")
            print(f"All Locations count: {len(locations)}")
            print("===============================")
            
            if not self.gemini_key:
                raise Exception("Gemini API key not configured")
            
            if not self.gemini_model:
                raise Exception("Gemini model not initialized")
            
            # Create the prompt
            prompt = self.create_gemini_prompt(preferences, decisions, locations)
            
            # Define the JSON schema for the response
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "tripTitle": {
                            "type": "string",
                            "description": "A creative and catchy title for the trip"
                        },
                        "days": {
                            "type": "array",
                            "description": "An array of daily itinerary plans",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date": {"type": "string", "description": "The date for this day's plan"},
                                    "day": {"type": "integer", "description": "The day number of the itinerary"},
                                    "theme": {"type": "string", "description": "A short theme for the day"},
                                    "events": {
                                        "type": "array",
                                        "description": "A list of events scheduled for the day",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "time": {"type": "string", "description": "The suggested time for the event"},
                                                "title": {"type": "string", "description": "The title of the event or activity"},
                                                "description": {"type": "string", "description": "A brief description of the event"},
                                                "type": {"type": "string", "description": "The type of event: activity, travel, food, or lodging"}
                                            },
                                            "required": ["time", "title", "description", "type"]
                                        }
                                    }
                                },
                                "required": ["date", "day", "theme", "events"]
                            }
                        }
                    },
                    "required": ["tripTitle", "days"]
                }
            )
            
            # Generate content with Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse the JSON response
            json_text = response.text.strip()
            parsed_json = json.loads(json_text)
            
            # Basic validation
            if parsed_json.get('tripTitle') and isinstance(parsed_json.get('days'), list):
                return {
                    'success': True,
                    'itinerary': parsed_json,
                    'method': 'gemini_v0_generation',
                    'planner_type': 'GeminiV0TravelPlanner'
                }
            else:
                raise ValueError("Generated JSON does not match the expected Itinerary structure")
                
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
        except Exception as e:
            print(f"Error generating itinerary: {str(e)}")
            raise Exception(f"Failed to generate itinerary: {str(e)}")


# Legacy function for backward compatibility
def generate_itinerary_v0(preferences: UserPreferences, decisions: List[SwipeDecision], locations: List[ProminentLocation]) -> Dict[str, Any]:
    """
    Legacy function that creates a temporary GeminiV0TravelPlanner instance.
    This is kept for backward compatibility.
    """
    # Get API key from environment
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    
    planner = GeminiV0TravelPlanner(gemini_key=GEMINI_API_KEY)
    return planner.generate_itinerary(preferences, decisions, locations)