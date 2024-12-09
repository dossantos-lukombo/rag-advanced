# from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.schema import Document
from typing import List
import warnings,os,dotenv
warnings.filterwarnings("ignore")

dotenv.load_dotenv(".env")

async def KG_document_generator(input_file:list[str],user_query:str):

    slm = ChatOllama(temperature=0, model="llama3.2")
    # llm_transformer = LLMGraphTransformer(llm=llm)

    print("Building Neo4j Vector Index...")
    print("Loading Documents...")
    print("Input File: ",input_file)
   
    vector_index = Neo4jVector(
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        url=os.getenv("URL"),
        username=os.getenv("USERNAME"),
        password=os.getenv("PASSWORD"),
        pre_delete_collection=True,
    ).from_documents(
        url=os.getenv("URL"),
        username=os.getenv("USERNAME"),
        password=os.getenv("PASSWORD"),
        documents=input_manager(input_file),
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        distance_strategy="COSINE",
    )
    
    
    print("Neo4j Vector Index Built...")
    
    template_kg_prompt = """
        You are a assistant, you are given a user request and context. 
        You are to generate a response to the user request.\n\n
        user_request: \n{user_query}\n
        context: \n{context}\n
        """
    
    prompt_slm_retriever = PromptTemplate( 
        input_variables=["user_query"],
        template=template_kg_prompt,
    )
    
    runnable = (
        {
            "user_query":RunnablePassthrough(),
            "context":vector_index.as_retriever(),
        } |
        prompt_slm_retriever |
        slm |
        StrOutputParser()
    )
    
    ans = ""
    count_period = 0
    async for chunk in runnable.astream(input={"user_query":user_query}) :
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


def input_manager(input_file:list[str])->List[Document]:
        
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


