from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
import json
from torch.cuda import is_available 
from llmops import llmInteractions
load_dotenv()
device="cuda:0" if is_available() else "cpu"

llm=-1
app = FastAPI()
origins = [
    '*',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
async def ingestion(request:Request):
    data= await request.json()
    invalid=[]
    for i in data:
        if(type(data[i])!=list):
            invalid.append(i)
        else:
            data[i]=data[i][0]
    if(len(invalid)):
        response={"message":"Empty Fields Found. CANNOT INGEST","Invalid Fields":invalid}
    else:
        await llm.ingest(data)
        response={"message": "Data Entered Successfully"}
    return response

@app.post("/infer")
async def inference(request:Request):
    data= await request.json()
    response= await llm.inference(data['query'])
    return response


# Below paths are for debugging purposes 
@app.post("/changeDatabaseHost")
async def hhost(request:Request):
    data= await request.json()
    os.environ["MilvusHost"]=data['name']
    global llm
    try:
        llm=llmInteractions()
        response=f"<host>:<port> = {os.getenv('MilvusHost')}:{os.getenv('MilvusPort')}"
    except Exception as e:
        response=e
    return response
@app.post("/changeDatabasePort")
async def pport(request:Request):
    data= await request.json()
    os.environ["MilvusPort"]=data['name']
    global llm
    try:
        llm=llmInteractions()
        response=f"<host>:<port> = {os.getenv('MilvusHost')}:{os.getenv('MilvusPort')}"

    except Exception as e:
        response=e
  
    return response
@app.post("/changeGeminiKey")
async def pport(request:Request):
    data= await request.json()
    os.environ["GOOGLE_API_KE"]=data['name']
    global llm
    try:
        llm=llmInteractions()
        response=os.environ["GOOGLE_API_KE"]
    except Exception as e:
        # print(e)
        response=e
  
    return response
if __name__ == "__main__":

    try:
        
        llm=llmInteractions()
    except:
        pass
    uvicorn.run(
        app,
        host=os.getenv("ApiHost"),
        port=int(os.getenv("ApiPort")),
    )