# LLM-Project-Using-LangChain-Google-Palm
This project is about to create a FAQ chatbot using Google Palm and langchain.

## Required Libraries
    import os
    from langchain.llms import GooglePalm 
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA 
    from langchain.prompts import PromptTemplate
    from langchain.document_loaders.csv_loader import CSVLoader
    from dotenv import load_dotenv
    load_dotenv()

## Google PaLM 
    The API key for Google PaLM can be obtained from the Google MakerSuite website.

## Langchain 
    LangChain is an open-source framework designed to simplify the creation of applications using  large language models (LLMs)

## Word Embedding
InstructEmbeddings used from langchain.
    
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    instructor_embedding = HuggingFaceInstructEmbeddings()

    
## vector database
FAISS vector database is used from langchain.

    from langchain.vectorstores import FAISS
    
    def create_vector_DB():
      loader = CSVLoader(file_path = "FAQs.csv", source_column = "prompt" )
      data = loader.load()
      vectordb = FAISS.from_documents(documents = data, 
                                      embedding = instructor_embedding)
      vectordb.save_local(vectordb_file_path)

## Retriving data
    def get_qa_chain():
        vectordb = FAISS.load_local(vectordb_file_path, instructor_embedding)
        retriever = vectordb.as_retriever(score_threshold = 0.7)
        
    
        prompt_template = """Given the following context and a question, generate an answer based                               on this context only. In the answer try to provide as much text as                                 possible from "response" section in the source document context                                    without making many changes. If the answer is not found in the 
                            context, kindly state "I don't know." Don't try to make up an answer.
    
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

## User Interface
The user interface is created using streamlit.
    ![image](https://github.com/priyanthan07/LLM-Project-Using-LangChain-Google-Palm/assets/129021635/97ddb8e3-a75b-4292-8a9c-ef4ef72c41b8)

