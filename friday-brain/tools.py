"""
tools.py - Tool Initialization for CrewAI Agents

This module initializes external tools that agents can use.
Currently, it provides DuckDuckGo search for internet access.

To add more tools:
1. Import the tool from langchain_community.tools
2. Initialize it and add to the TOOLS dictionary
3. Reference it in agents.py
"""

from langchain_community.tools import DuckDuckGoSearchRun
from typing import Dict, Any

# Initialize DuckDuckGo Search Tool
# This tool allows agents to search the internet in real-time
search_tool = DuckDuckGoSearchRun()

# Dictionary of all available tools
TOOLS = {
    "search": search_tool,
}


def get_tool(tool_name: str) -> Any:
    """
    Retrieve a tool by name.
    
    Args:
        tool_name: The name of the tool (e.g., 'search')
    
    Returns:
        The tool object
    
    Raises:
        ValueError: If tool not found
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}")
    return TOOLS[tool_name]


def get_all_tools() -> list:
    """
    Get all initialized tools as a list for CrewAI agents.
    
    Returns:
        List of all tools
    """
    return list(TOOLS.values())
