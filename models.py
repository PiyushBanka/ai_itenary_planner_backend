"""
Shared data models for the travel itinerary planner.
This module contains Pydantic models that are used across different modules.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Coordinates(BaseModel):
    """Represents geographical coordinates."""
    latitude: float
    longitude: float


class ProminentLocation(BaseModel):
    """Represents a prominent location in a city."""
    id: str
    name: str
    description: str
    tags: List[str]  # e.g., ['museum', 'outdoor', 'history', 'art']
    city: str
    city_id: int
    area: str
    image: str
    coordinates: Coordinates
    # avg_visit_duration_hours: float = 2.0  # Average time spent at the location
    # ticket_cost: float = 20.0 # Average ticket cost in USD


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


class ItineraryRequest(BaseModel):
    """Request model for itinerary generation."""
    preferences: UserPreferences
    swipe_decisions: List[SwipeDecision]