from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware #CORS 예외 URL을 등록하여 해결

import numpy
from models import bag_of_words, chat


app = FastAPI()

# origins = [ #forntend 서버 url
#     "http://localhost:5173",    # 또는 "http://localhost:5173"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #허용할 url(출처)
    allow_credentials=True, #자격 증명(credentials)을 허용할지 여부
    allow_methods=["*"],    #허용할 HTTP 메서드를 설정
    allow_headers=["*"],    #허용할 HTTP 헤더를 설정
)

@app.get("/")
def home():
    return {"message": "Hello"}

@app.get("/test")
def hello(question: str):
    s = chat(question)

    return {"message": s}

# if __name__ == '__main__':
#     uvicorn.run('main:app', port=1557, host='0.0.0.0', reload=True)