from crewai_tools import PDFSearchTool
from crewai import Task, Crew, Process
from crewai import Agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()




load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Correct LLM Configuration
llm = ChatGoogleGenerativeAI(
    provider="google",  # Explicitly specify the provider
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    
)




# Corrected PDFSearchTool configuration
pdf_tool = PDFSearchTool(
    config=dict(
        llm=dict(
            provider="google",  # Correct provider for Google Gemini LLM
            config=dict(
                model="gemini-1.5-flash",  # Explicit model for Google Gemini
                temperature=0.5,  # Example optional parameter
                
            ),
        ),
        embedder=dict(
            provider="google",  # Correct provider for embedding service
            config=dict(
                model="models/embedding-001",  # Embedding model
                task_type="retrieval_document",  # Specific task type for embeddings
                
            ),
        ),
    )
)

# Define agents with backstories
rag_agent = Agent(
    role='RAG System',
    goal='Query the PDF document via RAG using Google Gemini.',
    backstory="You are an advanced retrieval-augmented generation system, skilled in analyzing and extracting information from documents.",
    tools=[pdf_tool],
    
)

summarizer_agent = Agent(
    role='Summarization Expert',
    goal='Summarize the PDF content into a concise report.',
    backstory="You are a language model trained to summarize large amounts of information into clear and concise formats.",
    tools=[],
    
)

risk_assessor_agent = Agent(
    role='Risk Assessor',
    goal='Analyze risks in the PDF and generate a risk report.',
    backstory="You specialize in identifying and evaluating potential risks from complex documents.",
    tools=[],
    
)




# Define tasks using the updated rag_agent
query_task = Task(
    description="Chat with the PDF via RAG.",
    expected_output="Detailed responses to user queries.",
    agent=rag_agent
)

summarize_task = Task(
    description="Summarize the PDF into a concise document.",
    expected_output="A summarized report of the PDF.",
    agent=summarizer_agent
)

risk_task = Task(
    description="Identify and explain risks described in the PDF.",
    expected_output="A detailed risk assessment report.",
    agent=risk_assessor_agent
)

# Crew definition
crew = Crew(
    agents=[rag_agent, summarizer_agent, risk_assessor_agent],
    tasks=[query_task, summarize_task, risk_task],
    process=Process.sequential
)

def menu():
    print("Select an option:")
    print("1. Chat with PDF")
    print("2. Summarize PDF")
    print("3. Risk Assessment Report")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    pdf_path = input("Enter the path to the PDF: ").strip()
    
    if choice == '1':
        query = input("Enter your query: ").strip()
        result = Crew(
            agents=[rag_agent],
            tasks=[query_task],
            process=Process.sequential
        ).kickoff(inputs={'pdf_path': pdf_path, 'query': query})
        print("Chat Result:")
        print(result)
    
    elif choice == '2':
        result = Crew(
            agents=[summarizer_agent],
            tasks=[summarize_task],
            process=Process.sequential
        ).kickoff(inputs={'pdf_path': pdf_path})
        print("Summary Result:")
        print(result)
    
    elif choice == '3':
        result = Crew(
            agents=[risk_assessor_agent],
            tasks=[risk_task],
            process=Process.sequential
        ).kickoff(inputs={'pdf_path': pdf_path})
        print("Risk Assessment Result:")
        print(result)
    
    else:
        print("Invalid choice. Please try again.")

# Main execution
if __name__ == "__main__":
    menu()


