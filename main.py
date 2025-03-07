import os
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import FAISS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from agent import llm

from agent import financial_agent

# Financial Report Analyst Agent
# Financial Report Analysis Task
class FinancialReportTask:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.FAISSretriever = None

    def setup_rag(self):
        """
        Set up RAG (Retrieval-Augmented Generation) for the PDF.
        """
        print("Loading PDF and setting up RAG...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        # Create embeddings using the LLM's embedding method
        embeddings = embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents, embeddings)
        self.FAISSretriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        retrieved_docs = self.FAISSretriever.invoke("Total Assets in ITC company?")
        print("RAG setup complete!")

    def answer_query(self, query):
        """
        Answer queries interactively using the RAG system.
        """
        if not self.FAISSretriever:
            raise ValueError("RAG system not set up. Call `setup_rag` first.")

        print(f"Answering query: {query}")
        #qa_chain = financial_agent.llm.create_qa_chain(self.retriever)
        #return qa_chain.run(query)


        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(self.FAISSretriever, question_answer_chain)    
        response = rag_chain.invoke({"input": "Provide A detail Summary Of this Annual Report ?"})
        return response['answer']
    


