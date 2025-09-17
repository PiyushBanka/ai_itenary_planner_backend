# AI Itinerary Planner Backend

This is the FastAPI backend for the AI Itinerary Planner application. It serves prominent location data and generates personalized itineraries using Google's Gemini AI.

## Setup

1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini AI API Key:**
   - Copy `.env.example` to `.env`
   - Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Replace `your_actual_gemini_api_key_here` with your actual API key in `.env`

## Running the Backend

### Option 1: Using the startup script
- **Windows**: Double-click `start.bat` or run `start.bat` in command prompt

### Option 2: Manual start
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: http://localhost:8000

## API Endpoints

### Location Endpoints
- `GET /` - Health check
- `GET /locations` - Get all prominent locations
- `GET /locations/city/{city_id}` - Get locations by city ID
- `GET /locations/tags?tags=tag1,tag2` - Get locations by tags (comma-separated)
- `GET /locations/{location_id}` - Get specific location by ID

### Itinerary Generation
- `POST /generate-itinerary` - Generate personalized itinerary using Gemini AI
  - **Request Body**: 
    ```json
    {
      "userPreferences": {...},
      "swipeDecisions": [...],
      "allLocations": [...]
    }
    ```
  - **Response**: Complete itinerary with daily plans and activities

## API Documentation

Once the server is running, you can view the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Database

The location data is stored in `database/prominent_locations.json`. This file contains all the prominent locations with their details, coordinates, and metadata.

## Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini AI API key (required for itinerary generation)

## Features

- **Real-time AI Itinerary Generation**: Uses Google's Gemini AI to create personalized travel itineraries
- **Location-based Recommendations**: Incorporates user's liked/super-liked locations
- **Flexible Preferences**: Supports various travel preferences, budgets, and group sizes
- **Smart Scheduling**: AI optimizes daily schedules based on geography and travel time
- **Family-friendly Options**: Adjusts recommendations based on presence of children