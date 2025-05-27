import gradio as gr
from data_loader import load_data_from_pdf, load_data_from_text
from model import ModelHandler
from config_handler import load_config
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.core.utils import truncate_text
from langchain_community.embeddings import FastEmbedEmbeddings
import os

# Load configuration and initialize ModelHandler
config = load_config("config.json")
# model_handler = ModelHandler(config)
path = None


def upload_and_process_pdf(file_path):
    # Process the uploaded PDF file and prepare it for querying.
    global path
    path = file_path
    print("Processing new PDF/TXT file:", file_path)
    # vector_store = load_data_from_pdf(file_path)
    return "PDF/TXT processed and data indexed successfully."


def get_file_type(file_path):
    # detect the file type

    _, file_extension = os.path.splitext(file_path)
    if file_extension:
        return file_extension[1:].lower()
    else:
        return 'Unknown'


def display_source_node(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> str:
    """Display source node"""
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        "Node ID: {} \n"
        "Score: {} \n"
        "Text: {} \n"
    ).format(source_node.node.node_id, source_node.score, source_text_fmt)
    if show_source_metadata:
        text_md += "Metadata: {} \n".format(source_node.node.metadata)
    if isinstance(source_node.node, ImageNode):
        text_md += "Image:"

    return text_md


def chat_response(model_trait, question):
    # Handle the chat response using the loaded model and context from the PDF.
    print('now file is in chat response')
    if path is None:
        return "No PDF/TXT has been processed yet. Please upload and process a PDF/TXT file first."
    type = get_file_type(path)

    try:
        '''
        load different types of files
        '''
        print(f"file type: {type}")
        if type == 'pdf':
            vector_store = load_data_from_pdf(path)
        else:
            vector_store = load_data_from_text(path)

        model_handler = ModelHandler(config, model_trait)

        service_context = ServiceContext.from_defaults(
            llm=model_handler.model, embed_model=FastEmbedEmbeddings()
        )

        retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                              search_kwargs={"k": 5, "score_threshold": 0.2})

        documents = SimpleDirectoryReader(input_dir=os.path.dirname(path)).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index_retriever = index.as_retriever(similarity_top_k=5)

        retrievals = index_retriever.retrieve(question)

        string = ""
        for n in retrievals:
            string += display_source_node(n, source_length=1500) + "\n"

        response = model_handler.get_response(question, retriever) + "\n\n\n" + string
        return response
    except Exception as e:
        return f"Error handling the query: {str(e)}"


def setup_gradio_interface():
    with gr.Blocks() as app:
        with gr.Column(scale=1):
            with gr.Row():
                file_input = gr.File(label="Upload PDF/TXT file")
                process_btn = gr.Button("Process PDF/TXT")
            status_label = gr.Label()  # Define a status label to display messages
            model_selector = gr.Dropdown(label="Choose model trait", choices=['llama2', 'llama3', 'llama4'])

        # Bind the status label to the output of the process button
        process_btn.click(upload_and_process_pdf, inputs=file_input, outputs=status_label)

        with gr.Column(scale=2):
            with gr.Row():
                query_input = gr.Textbox(label="Enter your question here")
                submit_query_btn = gr.Button("Submit")
            response_output = gr.Textbox(label="Response")

        # Bind the chat response function to the submit button
        submit_query_btn.click(chat_response, inputs=[model_selector, query_input], outputs=response_output)

        # Layout the elements
        gr.Markdown("## Upload and Process")
        gr.Markdown("## Ask a Question")
        gr.Markdown("## Response")

    return app


if __name__ == "__main__":
    app = setup_gradio_interface()
    app.launch()
