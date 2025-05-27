# gpu_ocr.py
import easyocr


class FastOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    def extract(self, img_path):
        return self.reader.readtext(img_path, detail=0)


# caching.py
from redis import Redis


class CacheManager:
    def __init__(self):
        self.redis = Redis()

    def get_embedding(self, text):
        if cached := self.redis.get(text):
            return cached
        # 计算并缓存...