# app.py
import logging
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from state_graph_definition import graph
import base64
import io
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# 配置 CORS 中间件
origins = [
    "http://localhost:12345",  # Vue 前端运行地址，按需调整
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
config = {"configurable": {"thread_id": "1"}}
@app.post("/chat", response_class=JSONResponse)
async def index_post(request: Request, user_input: str = Form(...)):
    messages = []
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                                  config,):
            for value in event.values():
                messages.append(value["messages"][-1].content)
    except Exception as e:
        logging.error(f"Error streaming messages: {e}")

    # try:
    #     logging.info("Attempting to generate graph image...")
    #     img_bytes = graph.get_graph().draw_mermaid_png()
    #     graph_image = base64.b64encode(img_bytes).decode('utf-8')
    #     logging.info("Graph image generated successfully.")
    # except Exception as e:
    #     logging.error(f"Error generating graph image: {e}")

    data = {
        "user_input": user_input,
        "messages": messages,
        #"graph_image": graph_image
    }
    return JSONResponse(content=data)