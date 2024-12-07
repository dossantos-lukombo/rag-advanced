from pymilvus import MilvusClient

def connect_to_milvus():
    milvus = MilvusClient()
    return milvus

def connect_to_db():
    milvus = connect_to_milvus()
    milvus._create_connection(uri="http://localhost:19530")
    # milvus.create_collection(
    #     collection_name="LangChainCollection",
    #     dimension=768,
    #     vector_field_name="embedding",
    #     metric_type="COSINE",
    #     auto_id=True
    #     )
    
    
connect_to_db()