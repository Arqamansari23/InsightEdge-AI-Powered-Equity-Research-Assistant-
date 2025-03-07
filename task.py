from crewai import Task

from agent import financial_agent



# Define the task
financial_task = Task(
    description="Analyze the annual financial report and answer questions interactively.",
    expected_output="An interactive RAG-based question-answering system for the PDF.",
    agent=financial_agent
)