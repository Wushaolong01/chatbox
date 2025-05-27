import os
import json
import logging
import tempfile
import gradio as gr
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.utils import truncate_text
from PIL import Image
import pytesseract
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_isd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# 环境配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型配置
MODEL_CONFIG = {
    "embedding": {
        "name": "BAAI/bge-small-en-v1.5",
        "cache_dir": "./models/embeddings"
    },
    "llm": {
        "name": "llama2",
        "temperature": 0.7,
        "num_ctx": 4096
    }
}

SUPPORTED_EXTS = [
    ".pdf", ".txt", ".docx", ".doc",
    ".xlsx", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg"
]

# 修改后的暗色系主题配置
custom_theme = gr.themes.Default(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    button_primary_background_fill="linear-gradient(90deg, #FF69B4 0%, #FF1493 100%)",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_hover="linear-gradient(90deg, #FF1493 0%, #FF6EB4 100%)",
    button_primary_border_color="#FF69B4",
    block_background_fill="#1A1A1A",
    block_label_background_fill="#2D2D2D",
    block_label_text_color="#FFB6C1",
    block_title_text_color="#FF69B4",
    body_background_fill="#121212",
    border_color_accent="#FF69B4",
    color_accent_soft="#4A1E35",
    input_background_fill="#2D2D2D",
    input_border_color="#4A1E35",
    shadow_spread="6px",
    slider_color="#FF69B4"
)

class MarkdownConverter:
    @staticmethod
    def convert_to_markdown(file_path: str) -> str:
        """将文档转换为Markdown格式"""
        try:
            elements = partition(filename=file_path, strategy="fast")
            return "\n\n".join(e.text for e in elements if hasattr(e, "text"))
        except Exception as e:
            logger.error(f"Markdown转换失败: {str(e)}")
            raise


class EmbeddingManager:
    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_embedder(cls):
        """初始化嵌入模型"""
        os.makedirs(MODEL_CONFIG["embedding"]["cache_dir"], exist_ok=True)
        return FastEmbedEmbeddings(
            model_name=MODEL_CONFIG["embedding"]["name"],
            cache_dir=MODEL_CONFIG["embedding"]["cache_dir"],
            threads=4
        )


class ImageProcessor:
    @staticmethod
    def extract_text_from_image(image_path, lang='chi_sim+eng'):
        """OCR文本提取"""
        try:
            img = Image.open(image_path)
            return pytesseract.image_to_string(img, config=f'--oem 3 --psm 6 -l {lang}').strip()
        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            raise


class DataProcessor:
    @staticmethod
    def _create_temp_file(content: str, suffix: str) -> str:
        """创建临时文件"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            return tmp.name

    @classmethod
    def _process_single_file(cls, file_path: str):
        """处理单个文件"""
        try:
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                content = f"# 图片内容\n\n{ImageProcessor.extract_text_from_image(file_path)}"
            else:
                content = MarkdownConverter.convert_to_markdown(file_path)

            tmp_path = cls._create_temp_file(content, ".md")
            loader = TextLoader(tmp_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=[
                    "\n\n# ", "\n## ", "\n### ", "\n\n", "\n"
                ]
            )
            pages = loader.load_and_split(text_splitter=text_splitter)

            for doc in pages:
                doc.metadata.update({
                    "source": file_path,
                    "file_type": os.path.splitext(file_path)[1][1:],
                    "converted": True
                })

            os.remove(tmp_path)
            return filter_complex_metadata(pages)
        except Exception as e:
            logger.error(f"文件处理失败 {file_path}: {str(e)}")
            raise

    @classmethod
    def process_files(cls, file_paths: List[str]) -> Chroma:
        """并行处理文件"""
        all_docs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(cls._process_single_file, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                try:
                    all_docs.extend(future.result())
                except Exception as e:
                    logger.error(f"处理失败 {futures[future]}: {str(e)}")
        return Chroma.from_documents(
            documents=all_docs,
            embedding=EmbeddingManager.get_embedder(),
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )


class ModelHandler:
    def __init__(self, config, trait):
        model_config = config["models"][trait]
        self.model = self._init_llm(model_config)
        self.prompt = PromptTemplate(
            template=model_config["prompt_template"],
            input_variables=["context", "question"]
        )
        self.chain = self._build_chain()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def _init_llm(self, model_config):
        """初始化大模型"""
        return ChatOllama(
            model=MODEL_CONFIG["llm"]["name"],
            temperature=MODEL_CONFIG["llm"]["temperature"],
            num_ctx=MODEL_CONFIG["llm"]["num_ctx"],
            base_url="http://localhost:11434",
            request_timeout=300
        )

    def _build_chain(self):
        """构建处理链"""
        return (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
        )

    def get_response(self, message, retriever):
        """获取响应"""
        try:
            docs = retriever.invoke(message)
            if not docs:
                return "⚠️ 未找到相关内容，请尝试调整问题或上传相关文档", []
            context = "\n\n".join(doc.page_content for doc in docs)
            return self.chain.invoke({"context": context, "question": message}), docs
        except Exception as e:
            logger.error(f"响应生成失败: {str(e)}")
            raise


class DocumentQAApp:
    def __init__(self):
        self.config = self._load_config()
        self.current_files = []
        self.vector_store = None
        self.index = None
        Settings.embed_model = LangchainEmbedding(EmbeddingManager.get_embedder())

    def _load_config(self):
        """加载配置文件"""
        try:
            with open("config.json") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"配置加载失败: {str(e)}")
            raise

    def process_files(self, file_paths: List[str]):
        """处理上传文件"""
        try:
            self.vector_store = DataProcessor.process_files(file_paths)
            self._build_index(file_paths)
            self.current_files = file_paths
            return f"✅ 已成功分析 {len(file_paths)} 个文档"
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            return f"❌ 处理失败: {str(e)}"

    def _build_index(self, file_paths: List[str]):
        """构建索引"""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                content = []
                for fp in file_paths:
                    if fp.lower().endswith((".png", ".jpg", ".jpeg")):
                        content.append(f"# 图片内容\n\n{ImageProcessor.extract_text_from_image(fp)}")
                    else:
                        content.append(MarkdownConverter.convert_to_markdown(fp))

                tmp_path = os.path.join(tmp_dir, "combined.md")
                with open(tmp_path, "w") as f:
                    f.write("\n\n".join(content))

                self.index = VectorStoreIndex.from_documents(
                    SimpleDirectoryReader(tmp_dir).load_data()
                )
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            raise

    def generate_response(self, model_trait, question):
        """生成回答"""
        if not self.vector_store or not self.index:
            return "⚠️ 请先上传并分析文档"

        try:
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "filter": {"source": {"$in": self.current_files}}
                }
            )
            model_handler = ModelHandler(self.config, model_trait)
            response, _ = model_handler.get_response(question, retriever)

            retrievals = self.index.as_retriever(similarity_top_k=4).retrieve(question)
            context = "\n".join(
                f"📌 参考内容 {i + 1}:\n{self._format_node(n)}\n"
                for i, n in enumerate(retrievals[:3])
            )

            return f"🤖 智能回答：\n{response}\n\n🔍 参考依据：\n{context}"
        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            return f"❌ 生成回答失败: {str(e)}"

    def _format_node(self, node, length=1000):
        """格式化结果"""
        content = node.node.get_content().strip()
        metadata = "\n".join(f"{k}: {v}" for k, v in node.node.metadata.items())
        return (
            f"相关性: {node.score:.3f}\n"
            f"元数据: {metadata}\n"
            f"内容: {truncate_text(content, length)}\n"
            "━━━━━━━━━━━━━━"
        )

    def launch(self):
        """启动界面"""
        css = """
        .gradio-container {
            max-width: 1200px !important;
            background: #121212 !important;
        }
        .dark {background: #121212}
        .prose {
            padding: 20px; 
            background: #2D2D2D;
            border-radius: 12px; 
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            border: 1px solid #4A1E35;
            color: #E0E0E0;
        }
        .file-upload {
            border: 2px dashed #FF69B4 !important; 
            padding: 20px;
            border-radius: 12px;
            background: #2D2D2D;
        }
        .status-box {
            background: #4A1E35; 
            padding: 15px; 
            border-radius: 8px;
            color: #FFB6C1;
        }
        .markdown h1, .markdown h2, .markdown h3 {
            color: #FF69B4 !important;
        }
        textarea, input {
            background: #2D2D2D !important;
            color: #E0E0E0 !important;
        }
        .dropdown {
            background: #2D2D2D !important;
            border-color: #4A1E35 !important;
        }
        """
        with gr.Blocks(title="Multiple2025 文档分析", theme=custom_theme, css=css) as app:
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #FF69B4; font-size: 2.5em; margin-bottom: 10px;">
                    Multiple2025 智能文档分析系统
                </h1>
                <p style="color: #B0B0B0; font-size: 1.1em;">上传您的文档/图片，获取深度语义分析结果</p>
            </div>
            """)


            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    gr.Markdown("### 第一步：上传文档", elem_classes="prose")
                    with gr.Group():
                        file_input = gr.File(
                            label="支持格式：PDF/DOC/PPT/XLS/图片等",
                            file_types=SUPPORTED_EXTS,
                            type="filepath",
                            file_count="multiple",
                            elem_classes="file-upload"
                        )
                        process_btn = gr.Button("开始分析文档", variant="primary")

                    gr.Markdown("### 分析状态", elem_classes="prose")
                    status = gr.Textbox(
                        label="处理进度",
                        interactive=False,
                        elem_classes="status-box"
                    )

                    gr.Markdown("### 分析模式", elem_classes="prose")
                    model_selector = gr.Dropdown(
                        label="选择分析模式",
                        choices=list(self.config["models"].keys()),
                        value="default"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 智能问答", elem_classes="prose")
                    question_input = gr.Textbox(
                        label="请输入问题",
                        placeholder="输入您需要分析的文档内容相关问题...",
                        lines=4,
                        max_lines=6
                    )
                    submit_btn = gr.Button("提交分析", variant="primary")

                    gr.Markdown("### 分析报告", elem_classes="prose")
                    response_output = gr.Markdown(
                        label="分析结果",
                        elem_classes="prose",
                        show_copy_button=True
                    )

            process_btn.click(
                self.process_files,
                inputs=file_input,
                outputs=status,
                api_name="process"
            )
            submit_btn.click(
                self.generate_response,
                inputs=[model_selector, question_input],
                outputs=response_output,
                api_name="analyze"
            )

        return app


if __name__ == "__main__":
    qa_app = DocumentQAApp()
    app = qa_app.launch()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        favicon_path="./multiple2025.ico"  # 请替换实际图标路径
    )