from crewai import Agent

from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))


# Define Agent using CrewAI
financial_agent = Agent(
    role="Financial Analyst",
    goal="Assist in analyzing annual financial reports and provide interactive insights.",
    backstory="Expert in financial data extraction and analysis, powered by advanced AI.",
    memory=True,
    llm=llm
)

