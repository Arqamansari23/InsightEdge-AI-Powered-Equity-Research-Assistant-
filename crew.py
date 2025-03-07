from crewai import Crew,Process
from agent import financial_agent
from task import financial_task

from main import FinancialReportTask


# Assemble the crew
crew = Crew(
    agents=[financial_agent],
    tasks=[financial_task],
    process=Process.sequential
)



# Kickoff the crew
def main():
    # Path to the PDF file
    pdf_path ="E:\\Mywork\\fyp\\Fyp_Agentic\\data\\Annual_Report_2023.pdf"

    
    # Create and setup the financial agent
    financial_report_agent = FinancialReportTask(pdf_path)
    financial_report_agent.setup_rag()

    print("Agent is ready! You can now ask questions about the report.")
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = financial_report_agent.answer_query(query)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()


