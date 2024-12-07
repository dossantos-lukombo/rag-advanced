import gradio as gr
from logic.rag import input_manager, generator,pipeline
    
def iFace():
    gr.Interface(
        fn=pipeline,
        inputs=[
            gr.File(label="Upload Files", type="filepath", file_count="multiple",file_types=[".pdf", ".txt"]),
            gr.Textbox(label="User Request", placeholder="Enter your request here..."), 
        ],
        outputs=gr.Textbox(label="Answer",autoscroll=True,type="text"),
        description="This is a chatbot that can answer your questions based on the uploaded files.",
        title="RAG Chatbot",
    ).launch()
    
iFace()

# with gr.Blocks() as demo:
#     with gr.Column():
#         with gr.Row():
#             file_input = gr.File(
#                 file_count="multiple",
#                 file_types=[".pdf", ".txt"],
#                 label="Upload Files"
#             )
            

#         chatbot = gr.Chatbot(
#             label="Chatbot Conversation",
#             height=400,
#             type="messages",
#             layout="bubble",
#         )
        
#         user_request = gr.Textbox(
#                 label="User Request",
#                 placeholder="Enter your request here..."
#             )

#         submit = gr.Button("Submit")

#         submit.click(
#             fn=handle_interaction,
#             inputs=[file_input, user_request, chatbot],
#             outputs=[chatbot]
#         )

# demo.launch()
