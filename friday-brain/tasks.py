"""
"""
tasks.py - Task Definitions for CrewAI

This module defines the specific tasks that agents will execute.
Each task maps to a specific agent and defines the work to be done.
"""

from crewai import Task


def create_research_task(user_command: str, research_agent) -> Task:
    """
    Create a research task for the Research Agent.
    
    This task gathers information from the internet using available tools
    to answer the user's command.
    
    Args:
        user_command: The user's input command
        research_agent: The research agent that will execute this task
    
    Returns:
        Task object for the research agent
    """
    return Task(
        description=f"""Analyze the following user command and gather relevant, up-to-date information.
        
User Command: {user_command}

Your objectives:
1. Identify the key information needed to respond to this command
2. Use available tools (especially internet search) to find current, relevant facts
3. Gather comprehensive and accurate information
4. Compile a detailed summary of your findings

Focus on accuracy and relevance to the user's request.""",
        agent=research_agent,
        expected_output="A comprehensive summary of research findings relevant to the user's command",
    )


def create_response_task(research_output: str, synthesizer_agent) -> Task:
    """
    Create a response task for the Synthesizer Agent.
    
    This task takes the research findings and formats them into a final,
    conversational response from FRIDAY.
    
    Args:
        research_output: The output from the research task
        synthesizer_agent: The synthesizer agent that will execute this task
    
    Returns:
        Task object for the synthesizer agent
    """
    return Task(
        description=f"""Take the following research findings and synthesize them into a clear,
conversational response that acts as FRIDAY addressing the user's command.

Research Findings:
{research_output}

Your objectives:
1. Extract the most important and relevant insights from the research
2. Format the response in a friendly, conversational tone as FRIDAY
3. Make the information clear and easy to understand
4. Directly address the user's command with actionable insights
5. Keep the response concise but comprehensive

Deliver a final response that sounds like FRIDAY, an intelligent and helpful AI assistant.""",
        agent=synthesizer_agent,
        expected_output="A final, conversational response from FRIDAY that addresses the user's command",
    )
