import gradio as gr
from logic.kg_rag import KG_document_generator
    
def iFace():
    gr.Interface(
        fn=KG_document_generator,
        inputs=[
            gr.File(label="Upload Files", type="filepath", file_count="multiple",file_types=[".pdf",".csv", ".txt"]),
            gr.Textbox(label="User Request", placeholder="Enter your request here..."), 
        ],
        outputs=gr.Textbox(label="Answer",autoscroll=True,type="text"),
        description="This is a chatbot that can answer your questions based on the uploaded files.",
        title="RAG Chatbot",
    ).launch()
