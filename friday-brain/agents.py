"""
agents.py - CrewAI Agent Definitions

This module defines the agents that will process commands:
- Research_Agent: Gathers information using tools
- Execution_Agent: Formats and delivers the final answer

To swap LLM providers:
1. Change the 'model' parameter in the Agent initialization
2. For OpenAI (default): model="gpt-4" or "gpt-3.5-turbo"
3. For Ollama (local): model="ollama/neural-chat" or any pulled model
4. Ensure your API keys are set in .env file
"""

from crewai import Agent
from tools import get_all_tools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get all available tools
tools = get_all_tools()

# ============================================================================
# LLM Configuration Examples
# ============================================================================
# 
# OPENAI (Default):
# - Requires: OPENAI_API_KEY in .env
# - Models: "gpt-4", "gpt-3.5-turbo"
# 
# OLLAMA (Local LLM):
# - Requires: Ollama running locally on port 11434
# - Models: "neural-chat", "mistral", "dolphin-phi", etc.
# - Setup: ollama pull <model-name>
# - Change model parameter to use different models
#
# ANTHROPIC (Claude):
# - Requires: ANTHROPIC_API_KEY in .env
# - Models: "claude-3-opus", "claude-3-sonnet"
#
# ============================================================================

# Research Agent - Gathers factual information
research_agent = Agent(
    role="Research Agent",
    goal="Find accurate and relevant information to answer the user's command",
    backstory="""You are a thorough research specialist. Your role is to search
    for accurate, up-to-date information and facts to support the user's request.
    You use available tools to gather information from the internet and synthesize
    it into useful insights.""",
    tools=tools,
    verbose=True,
    # ========================================================================
    # LLM Provider Configuration
    # ========================================================================
    # For OpenAI (default):
    # llm_provider="openai",
    # model="gpt-4",  # or "gpt-3.5-turbo"
    
    # For Ollama (uncomment to use local LLM):
    # llm_provider="ollama",
    # model="neural-chat",
    
    # For Anthropic Claude (uncomment to use):
    # llm_provider="anthropic",
    # model="claude-3-opus",
    
    # Default behavior: Uses OPENAI_API_KEY from environment
    allow_delegation=False,
    memory=True,  # Enable memory to remember context across tasks
)

# Synthesizer Agent - Formats and delivers the final FRIDAY response
synthesizer_agent = Agent(
    role="Synthesizer Agent",
    goal="Format research findings into a clear, conversational response acting as FRIDAY",
    backstory="""You are FRIDAY's communication specialist. Your job is to take the
    research and information gathered and transform it into a clear, concise,
    and conversational response that directly addresses the user's command.
    You represent FRIDAY, a helpful and knowledgeable AI assistant.""",
    verbose=True,
    # ========================================================================
    # LLM Provider Configuration (must match research_agent for consistency)
    # ========================================================================
    # For OpenAI (default):
    # llm_provider="openai",
    # model="gpt-4",  # or "gpt-3.5-turbo"
    
    # For Ollama (uncomment to use local LLM):
    # llm_provider="ollama",
    # model="neural-chat",
    
    # For Anthropic Claude (uncomment to use):
    # llm_provider="anthropic",
    # model="claude-3-opus",
    
    allow_delegation=False,
    memory=True,  # Enable memory to remember context across tasks
)


def get_agents() -> tuple:
    """
    Retrieve the configured agents.
    
    Returns:
        Tuple of (research_agent, synthesizer_agent)
    """
    return research_agent, synthesizer_agent
