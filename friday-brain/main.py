"""
main.py - FastAPI Application for the FRIDAY AI Brain

This is the main entry point for the AI Brain API.
It exposes a POST /process-command endpoint that receives text commands,
processes them through CrewAI agents, and returns structured JSON responses.

The server uses async/await to prevent blocking while agents are processing.

To run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

To test:
    curl -X POST http://localhost:8000/process-command \\
         -H "Content-Type: application/json" \\
         -d '{"text": "What is the current weather in San Francisco?"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import CrewAI components
from crewai import Crew, Task, Agent
from agents import get_agents
from tasks import create_research_task, create_response_task

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App Initialization
# ============================================================================
app = FastAPI(
    title="FRIDAY AI Brain API",
    description="A headless AI assistant backend using CrewAI and FastAPI",
    version="1.0.0",
)

# ============================================================================
# Pydantic Models for Input/Output Validation
# ============================================================================

class CommandRequest(BaseModel):
    """Input schema for the /ask-friday endpoint"""
    command: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's text command or query for FRIDAY"
    )

    class Config:
        example = {
            "command": "What are the latest developments in AI?"
        }


class CommandResponse(BaseModel):
    """Output schema for the /ask-friday endpoint"""
    status: str = Field(
        description="Status of the request: 'success' or 'error'"
    )
    response: str = Field(
        description="FRIDAY's final response to the user's command"
    )

    class Config:
        example = {
            "status": "success",
            "response": "The latest developments in AI include..."
        }


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        JSON with status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "FRIDAY AI Brain"
    }


# ============================================================================
# Main Command Processing Endpoint
# ============================================================================

@app.post("/ask-friday", response_model=CommandResponse)
async def ask_friday(request: CommandRequest):
    """
    Main endpoint for asking FRIDAY questions or giving commands.
    
    This endpoint:
    1. Validates the user's command
    2. Instantiates a Crew with research and synthesizer agents
    3. Processes the command through the agentic workflow
    4. Returns FRIDAY's response as JSON
    
    Args:
        request: CommandRequest containing the user's command
    
    Returns:
        CommandResponse with status and FRIDAY's response
    
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"FRIDAY received command: {request.command[:100]}...")
        
        # Get the configured agents
        research_agent, synthesizer_agent = get_agents()
        
        # Create tasks for this specific command
        research_task = create_research_task(request.command, research_agent)
        response_task = create_response_task(
            "{{ research task output }}",  # CrewAI injects previous task output
            synthesizer_agent
        )
        
        # Create a Crew with the agents and tasks
        # The Crew orchestrates the workflow between agents
        crew = Crew(
            agents=[research_agent, synthesizer_agent],
            tasks=[research_task, response_task],
            verbose=True,
        )
        
        # ====================================================================
        # Async Processing - Prevents blocking the event loop
        # ====================================================================
        # Run the crew in a thread pool to avoid blocking FastAPI's async loop
        # This allows multiple concurrent requests to be handled efficiently
        
        loop = asyncio.get_event_loop()
        
        # Execute the crew's kickoff process in a thread pool
        # crew.kickoff() runs the entire workflow and returns the final output
        result = await loop.run_in_executor(
            None,
            crew.kickoff
        )
        
        logger.info("FRIDAY successfully processed the command")
        
        # ====================================================================
        # Response Construction
        # ====================================================================
        # Return the final response in the expected format
        response = CommandResponse(
            status="success",
            response=str(result)  # Convert the crew output to string
        )
        
        return response
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(ve)}"
        )
    
    except TimeoutError:
        logger.error("Command processing timed out")
        raise HTTPException(
            status_code=504,
            detail="Command processing timed out. Please try a simpler query."
        )
    
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


# ============================================================================
# Information Endpoints
# ============================================================================

@app.get("/info")
async def get_info():
    """
    Get information about the API and its configuration.
    
    Returns:
        JSON with API metadata
    """
    return {
        "name": "FRIDAY AI Brain",
        "version": "1.0.0",
        "description": "A headless AI assistant backend using CrewAI",
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check"
            },
            {
                "path": "/ask-friday",
                "method": "POST",
                "description": "Ask FRIDAY a question or give a command"
            },
            {
                "path": "/info",
                "method": "GET",
                "description": "Get API information"
            }
        ],
        "openapi_url": "/openapi.json",
        "docs_url": "/docs"
    }


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Called when the FastAPI application starts.
    Use this for initialization tasks.
    """
    logger.info("FRIDAY AI Brain API is starting up...")
    logger.info(f"OpenAI API Key configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    logger.info("Visit http://localhost:8000/docs for interactive API documentation")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Called when the FastAPI application shuts down.
    Use this for cleanup tasks.
    """
    logger.info("FRIDAY AI Brain API is shutting down...")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint with welcome message and usage instructions.
    """
    return {
        "message": "Welcome to FRIDAY AI Brain",
        "documentation": "Visit /docs for interactive API documentation",
        "health": "/health",
        "main_endpoint": "/ask-friday",
        "info": "/info"
    }


# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    """
    To run this application:
    
    1. Install dependencies:
       pip install -r requirements.txt
    
    2. Create a .env file with your API keys:
       OPENAI_API_KEY=your_key_here
       
       OR for Ollama:
       OLLAMA_BASE_URL=http://localhost:11434
    
    3. Run the server:
       uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
    4. Access the API:
       - Interactive docs: http://localhost:8000/docs
       - API root: http://localhost:8000/
       - Health: http://localhost:8000/health
       - Ask FRIDAY: POST http://localhost:8000/ask-friday
    
    5. Example curl request:
       curl -X POST http://localhost:8000/ask-friday \\
            -H "Content-Type: application/json" \\
            -d '{"command": "What are the latest developments in AI?}'
    """
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
