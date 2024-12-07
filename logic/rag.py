from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.milvus import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableSerializable,Runnable
from langchain_ollama import ChatOllama
from langchain.schema import Document
from typing import List
import asyncio
import warnings
warnings.filterwarnings("ignore")



def input_manager(input_file:list[str])->List[Document]:
    
    # print(input_file)
    
    extension:str = ""
    pdf_document_list:List[Document] = []
    
    for file in input_file:
        
        extension = file.split('.')[-1]
        if extension in ['pdf']:
            pdf_document_list=PDFPlumberLoader(file).load()
        if extension in ['txt']:
            with open(file, 'r') as f:
                pdf_document_list = [Document(f.read())]
        
    return pdf_document_list

async def pipeline(input_file:list[str], user_request:str):
    """
    pdf -> chunk -> embedding -> milvus -> slm -> output
    user_request -> slm -> embedding -> milvus -> slm -> output
    Pipeline: of the rag with two slm
    """
    
    extension:str = ""
    pdf_document_list:List[Document] = []
    
    for file in input_file:
        
        extension = file.split('.')[-1]
        if extension in ['pdf']:
            pdf_document_list=PDFPlumberLoader(file).load()
        if extension in ['txt']:
            with open(file, 'r') as f:
                pdf_document_list = [Document(f.read())]
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n'],
        )
    # try:
    #     text_splitter._chunk_size = int(Input_Analyser(user_request))
    # except int(Input_Analyser(user_request)) :
    text_splitter._chunk_size = 512
        
    print("chunk size : ",text_splitter._chunk_size)
   
    documents_split = text_splitter.split_documents(pdf_document_list)
    text_documents_split:List[str] = [doc.page_content for doc in documents_split]
    
    print("text_documents_split length : ",len(text_documents_split))
    
    embedding_model = OllamaEmbeddings(
        model = "nomic-embed-text",
    )
    
    vectore_store = Milvus(
            embedding_function=embedding_model,
            ).from_texts(
            texts=text_documents_split,
            embedding=embedding_model,
            collection_name="LangChainCollection",
            drop_old=True,
            index_params = {
                "metric_type": "COSINE",
            },
            search_params={
                "nprobe": 16,
                "nlist": 16384,
            }
        )
    
    slm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    
    
    template_slm_retriever = """
        You are a assistant, you are given a user request and context. You are to generate a response to the user request.\n\n
        user_request: {user_request}\n\n
        context: \n{context}\n
        """
    
    prompt_slm_retriever = PromptTemplate(
        input_variables=["user_request"],
        template=template_slm_retriever
    )
    
    runnable = (
        {
            "user_request":RunnablePassthrough(),
            "context":vectore_store.as_retriever(),
        } |
        prompt_slm_retriever |
        slm |
        StrOutputParser()
    )
    
    # runnable = RunnableSerializable(runnable)
    ans = ""
    count_period = 0
    async for chunk in runnable.astream(input={"user_request":user_request}) :
        ans += chunk
        if chunk == "." or chunk == "?" or chunk == "!" or chunk.count(".") > 0 :
            count_period += 1
        if count_period == 1 :
            ans += "\n"
            count_period += 1
        if count_period > 2 :
            ans += "\n\n"
            count_period = 0
        yield ans
    
    
async def generator(input_file, user_request:str):
    return pipeline(input_manager(input_file), user_request)
    
    
def Input_Analyser(user_request:str):
    
    
    slm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    
    template_slm_request = """
    Analyze the given user request and determine the optimal chunk size (in number of words) 
    to ensure good performance for Retrieval-Augmented Generation (RAG). 
    You must Provide only a single number as your answer, without any additional explanation or text.\n\n
    User request: {user_request}
    """
    
    slm_request = PromptTemplate(
        input_variables=["user_request"],
        template=template_slm_request
    )
    
    runnable = (
        {
            "user_request":RunnablePassthrough(),
        } |
        slm_request |
        slm |
        StrOutputParser()
    )
    
    output_response=runnable.invoke(input={"user_request":user_request})
        
    return output_response
 