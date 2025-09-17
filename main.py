from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Itinerary Planner API", version="1.0.0")

# Configure Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
from pydantic import BaseModel

class Coordinates(BaseModel):
    lat: float
    lng: float

class ProminentLocation(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    city: str
    area: str
    image: str
    coordinates: Coordinates
    city_id: int

# Itinerary generation models
class UserPreferences(BaseModel):
    location: str
    numberOfAdults: int
    numberOfChildren: int
    ticketsBooked: bool
    # Fields for when tickets ARE booked
    arrivalDateTime: Optional[str] = None
    arrivalLocation: Optional[str] = None
    departureDateTime: Optional[str] = None
    departureLocation: Optional[str] = None
    # Fields for when tickets are NOT booked
    startingCity: Optional[str] = None
    leaveHomeDateTime: Optional[str] = None
    returnHomeDateTime: Optional[str] = None
    localTransport: str  # 'car rental' | 'bike rental' | 'public transport'
    hotelQuality: str    # 'hostel' | 'budget' | 'standard' | 'luxury' | 'boutique'
    poolRequired: bool
    budget: int
    interests: List[str]
    tripDescription: str

class SwipeDecision(BaseModel):
    locationId: str
    choice: str  # 'left' | 'right' | 'superlike'

class ItineraryEvent(BaseModel):
    time: str
    title: str
    description: str
    type: str

class ItineraryDay(BaseModel):
    date: str
    day: int
    theme: str
    events: List[ItineraryEvent]

class Itinerary(BaseModel):
    tripTitle: str
    days: List[ItineraryDay]

class ItineraryRequest(BaseModel):
    userPreferences: UserPreferences
    swipeDecisions: List[SwipeDecision]
    allLocations: List[ProminentLocation]

# Global variable to store locations data
prominent_locations: List[ProminentLocation] = []

def load_locations_data():
    """Load prominent locations from JSON file"""
    global prominent_locations
    database_path = Path(__file__).parent / "database" / "prominent_locations.json"
    
    try:
        if database_path.exists() and database_path.stat().st_size > 0:
            with open(database_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                prominent_locations = [ProminentLocation(**location) for location in data]
        else:
            prominent_locations = []
    except Exception as e:
        print(f"Error loading locations data: {e}")
        prominent_locations = []

# Load data on startup
load_locations_data()

@app.on_event("startup")
async def startup_event():
    """Load data when the application starts"""
    load_locations_data()
    print(f"Loaded {len(prominent_locations)} prominent locations")

@app.get("/")
async def root():
    return {"message": "AI Itinerary Planner API", "status": "running"}

@app.get("/locations", response_model=List[ProminentLocation])
async def get_all_locations():
    """Get all prominent locations"""
    return prominent_locations

@app.get("/locations/city/{city_id}", response_model=List[ProminentLocation])
async def get_locations_by_city(city_id: int):
    """Get locations by city ID"""
    city_locations = [loc for loc in prominent_locations if loc.city_id == city_id]
    if not city_locations:
        raise HTTPException(status_code=404, detail=f"No locations found for city_id {city_id}")
    return city_locations

@app.get("/locations/tags", response_model=List[ProminentLocation])
async def get_locations_by_tags(tags: str):
    """Get locations by tags (comma-separated)"""
    tag_list = [tag.strip() for tag in tags.split(",")]
    filtered_locations = []
    
    for location in prominent_locations:
        if any(tag in location.tags for tag in tag_list):
            filtered_locations.append(location)
    
    return filtered_locations

@app.get("/locations/{location_id}", response_model=ProminentLocation)
async def get_location_by_id(location_id: str):
    """Get a specific location by ID"""
    for location in prominent_locations:
        if location.id == location_id:
            return location
    raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found")

# Helper functions for itinerary generation
def get_location_name_by_id(location_id: str, locations: List[ProminentLocation]) -> str:
    """Get location name by ID"""
    location = next((loc for loc in locations if loc.id == location_id), None)
    return location.name if location else 'Unknown Location'

def create_gemini_prompt(preferences: UserPreferences, decisions: List[SwipeDecision], locations: List[ProminentLocation]) -> str:
    """Create the prompt for Gemini AI"""
    
    # Extract liked locations based on choice field
    super_liked = [get_location_name_by_id(d.locationId, locations) 
                   for d in decisions if d.choice == 'superlike']
    liked = [get_location_name_by_id(d.locationId, locations) 
             for d in decisions if d.choice == 'right']
    
    travel_prompt_section = ''
    task_structure_prompt = ''
    
    if preferences.ticketsBooked and preferences.arrivalDateTime and preferences.departureDateTime:
        from datetime import datetime
        try:
            # Handle different datetime formats
            arrival_str = preferences.arrivalDateTime
            departure_str = preferences.departureDateTime
            
            # Remove Z and handle timezone
            if arrival_str.endswith('Z'):
                arrival_str = arrival_str[:-1] + '+00:00'
            if departure_str.endswith('Z'):
                departure_str = departure_str[:-1] + '+00:00'
                
            arrival = datetime.fromisoformat(arrival_str)
            departure = datetime.fromisoformat(departure_str)
            duration_ms = (departure - arrival).total_seconds() * 1000
            duration_days = max(1, int((duration_ms / (1000 * 60 * 60 * 24)) + 0.5))
            
            travel_prompt_section = f"""
            - **Travel Plan:** Tickets are already booked.
            - **Arrival:** The user will arrive at {preferences.arrivalLocation} on {arrival.strftime('%A, %B %d, %Y at %I:%M %p')}.
            - **Departure:** The user will depart from {preferences.departureLocation} on {departure.strftime('%A, %B %d, %Y at %I:%M %p')}.
            - **Trip Duration:** The total duration is {duration_days} days.
            """
            task_structure_prompt = f"Create a plan for each of the {duration_days} days. Use the provided arrival date to generate realistic dates for the itinerary."
        except Exception as e:
            print(f"Error parsing booked ticket dates: {e}")
            task_structure_prompt = "Create a 3-day itinerary as a default."
        
    elif preferences.startingCity and preferences.leaveHomeDateTime and preferences.returnHomeDateTime:
        from datetime import datetime
        try:
            leave_home_str = preferences.leaveHomeDateTime
            return_home_str = preferences.returnHomeDateTime
            
            # Remove Z and handle timezone
            if leave_home_str.endswith('Z'):
                leave_home_str = leave_home_str[:-1] + '+00:00'
            if return_home_str.endswith('Z'):
                return_home_str = return_home_str[:-1] + '+00:00'
                
            leave_home = datetime.fromisoformat(leave_home_str)
            return_home = datetime.fromisoformat(return_home_str)
            
            travel_prompt_section = f"""
            - **Travel Plan:** Tickets are NOT booked. The itinerary must include suggestions for travel to and from the destination.
            - **Origin:** The user is starting their journey from {preferences.startingCity}.
            - **Availability Window:** They can leave their home city anytime after {leave_home.strftime('%A, %B %d, %Y at %I:%M %p')} and must be back in their home city by {return_home.strftime('%A, %B %d, %Y at %I:%M %p')}.
            """
            task_structure_prompt = f"Create an itinerary that fits within this window, including reasonable travel time to and from {preferences.location}. The duration should be based on the user's availability and preferences."
        except Exception as e:
            print(f"Error parsing flexible travel dates: {e}")
            task_structure_prompt = "Create a 3-day itinerary as a default."
    else:
        task_structure_prompt = "Create a 3-day itinerary as a default."
    
    prompt = f"""
    You are an expert travel agent creating a personalized itinerary for a trip to {preferences.location}.

    **User Preferences:**
    - **Destination:** {preferences.location}
    - **Travelers:** {preferences.numberOfAdults} adults and {preferences.numberOfChildren} children.
    {travel_prompt_section}
    - **Local Transport:** The user prefers to use {preferences.localTransport}.
    - **Accommodation:** A {preferences.hotelQuality} quality hotel. A pool is {'required' if preferences.poolRequired else 'not required'}.
    - **Budget:** Approximately INR {preferences.budget} per person for the entire trip.
    - **Primary Interests:** {', '.join(preferences.interests) if preferences.interests else 'General sightseeing'}.
    - **User's Trip Description:** "{preferences.tripDescription or 'No specific description provided.'}"

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

@app.post("/generate-itinerary", response_model=Itinerary)
async def generate_itinerary(request: ItineraryRequest):
    """
    Generate an itinerary using Gemini AI based on user preferences, swipe decisions, and available locations
    """
    try:
        print("=== DEBUG: Received request ===")
        print(f"User Preferences: {request.userPreferences}")
        print(f"Swipe Decisions: {request.swipeDecisions}")
        print(f"All Locations count: {len(request.allLocations)}")
        print("===============================")
        
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        # Create the prompt
        prompt = create_gemini_prompt(request.userPreferences, request.swipeDecisions, request.allLocations)
        
        # Configure the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Parse the JSON response
        json_text = response.text.strip()
        parsed_json = json.loads(json_text)
        
        # Basic validation
        if parsed_json.get('tripTitle') and isinstance(parsed_json.get('days'), list):
            return Itinerary(**parsed_json)
        else:
            raise ValueError("Generated JSON does not match the expected Itinerary structure")
            
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        print(f"Error generating itinerary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate itinerary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)