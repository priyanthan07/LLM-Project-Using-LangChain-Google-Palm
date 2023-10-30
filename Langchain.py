import os
from langchain.llms import GooglePalm 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
load_dotenv()


llm = GooglePalm(google_api_key = os.environ["Google_API_Key"], temperature= 0.1)


instructor_embedding = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_DB():
    loader = CSVLoader(file_path = "FAQs.csv", source_column = "prompt" )
    data = loader.load()
    vectordb = FAISS.from_documents(documents = data, 
                                    embedding = instructor_embedding)
    vectordb.save_local(vectordb_file_path)
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embedding)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate(
        template = prompt_template , input_variables = ["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm =llm, chain_type = "stuff", 
        retriever = retriever, input_key = "query",
        return_source_documents = True , 
        chain_type_kwargs = {"prompt": prompt}
    )
    return chain

if __name__ == "__main__":
    #create_vector_DB()
    chain = get_qa_chain()
    print(chain("how to learn deep learning?"))