# hybrid_retriever.py
from elasticsearch import Elasticsearch
from FlagEmbedding import FlagReranker


class HybridRetriever:
    def __init__(self):
        self.es = Elasticsearch()
        self.reranker = FlagReranker('BAAI/bge-reranker-base')

    def reciprocal_rank_fusion(self, results_a, results_b, k=60):
        # 实现RRF算法...
        return fused_results

    def search(self, query):
        vector_results = Chroma().similarity_search(query, k=20)
        keyword_results = self.es.search(query, size=20)['hits']
        fused = self.reciprocal_rank_fusion(vector_results, keyword_results)
        return self.reranker.rerank(query, fused)[:10]


# smart_splitter.py
class SmartSplitter:
    def split_by_type(self, content, file_type):
        if file_type == "pdf":
            return self._split_with_headings(content)
        elif file_type == "code":
            return self._split_code(content)
        # 其他类型处理...