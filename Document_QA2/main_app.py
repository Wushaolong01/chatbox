# 保留原始DocumentQAApp类，修改以下部分：
class DocumentQAApp:
    def __init__(self):
        # 替换原有组件
        self.retriever = HybridRetriever()
        self.ocr = FastOCR()
        self.llm_router = ModelRouter()

    def process_files(self, files):
        # 使用async_process替代原处理逻辑
        tasks = [async_process.delay(f) for f in files]
        # 添加监控埋点
        Monitor().log_processing()

    def generate_response(self, question):
        # 使用新的检索和路由逻辑
        context = self.retriever.search(question)
        model = self.llm_router.select(question)
        return model.generate(context, question)