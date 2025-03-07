from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from fpdf import FPDF
from dotenv import load_dotenv
import os
load_dotenv()

def extract(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, 
        chunk_overlap=1000
    )
    texts = text_splitter.split_documents(pages)
    return texts

def summarize_document_with_kmeans_clustering(file, llm, embeddings):
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=10)
    texts = extract(file)
    
    try:
        result = filter.transform_documents(documents=texts)
        
        checker_chain = load_summarize_chain(llm, chain_type="refine")

        prompt =  prompt = """Generate a **detailed, structured, and multi-page** summary of the document. 
        The summary should be divided into the following sections:

        1️⃣ **Business Strategy** - Company's vision, expansion plans, market positioning.  
        2️⃣ **Financial Performance & Key Ratios** - Revenue growth, profitability, important ratios.  
        3️⃣ **Management Discussion & Future Outlook** - CEO's insights, company goals, industry trends.  
        4️⃣ **Governance & Risk Management** - Compliance, ethical practices, financial and operational risks.  
        5️⃣ **Shareholder Insights** - Major investors, dividends, stock performance, ownership details.  

        The summary should be **detailed (3-4 pages), formatted well**, and structured with clear section headings, with Numbers also like (Net Profit , dividends , Fimnancial ratios and all performance matrices number should be covered)."""
        summary = checker_chain.invoke({"input_documents": result, "question": prompt})
        return summary["output_text"]
    except Exception as e:
        return str(e)

class PDF(FPDF):
    def header(self):
        """Creates a styled header"""
        self.set_font("Times", "B", 16)
        self.cell(200, 10, "Annual Report Summary", ln=True, align="C")  # Title centered
        self.ln(5)

    def footer(self):
        """Creates a footer with page numbers"""
        self.set_y(-15)
        self.set_font("Times", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def save_summary_to_pdf(summary, output_file="summary.pdf"):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Times", "", 12)

    sections = [
        "Business Strategy",
        "Financial Performance & Key Ratios",
        "Management Discussion & Future Outlook",
        "Governance & Risk Management",
        "Shareholder Insights"
    ]

    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Annual Report Summary", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Times", "", 12)
    pdf.multi_cell(0, 6, summary.encode("latin-1", "replace").decode("latin-1"))
    
    pdf.output(output_file)
    print(f"✅ PDF saved successfully as {output_file}")

# Use Google Gemini for embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Use Google Gemini for LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0
)

summary = summarize_document_with_kmeans_clustering("ECORP-Annual-Report-2023-WEB-FINAL.pdf", llm, embeddings)
save_summary_to_pdf(summary)

