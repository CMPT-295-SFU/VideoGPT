import asyncio
import json
import logging
import os
import sys
from typing import AsyncIterable
import pandas as pd
import pinecone
import streamlit as st
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from Ask import (display_audio_results, display_slide_results,
                 get_audio_content, get_slide_content)
from components.sidebar import sidebar
import markdown

# set to DEBUG for more verbose logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key  = os.getenv("OPENAI_API_KEY")
# s3 = S3("classgpt")

def init_logger():
    logger.add("295API.log", level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message} | {extra[user]}", rotation="10 MB", compression="gz")


#client = OpenAI(api_key="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")
client = OpenAI(api_key=openai_api_key)
index_id = "295-youtube-index"
pinecone.init(
        api_key=pinecone_api_key,
        environment="us-west1-gcp-free"
    )
index = pinecone.Index(index_id)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/topic")
async def topic(query: str = None):
    print(query)
    # encoded_query = client.embeddings.create(input=query,  model="text-embedding-ada-002")['data'][0]['embedding']
    # res = query_gpt(chosen_class, chosen_pdf, query)
    text = query.replace("\n", " ")
    encoded_query = (
                client.embeddings.create(input=[text], model="text-embedding-ada-002")
                .data[0]
                .embedding
    )
    context = "Question: " + query + "\n"
    context += "\n" + "#######Slide Context#####\n"
    slide_results,content = get_slide_content(index, encoded_query) 
    context += "\n #######Audio Context#####\n"
    audio_results,content = get_audio_content(index, encoded_query)
    context += content            
    elements = []
    # Create json from list
    url = []
    # print(slide_results)
    for m in slide_results['results'][0]['matches']:
        url.append(m['metadata']['file'])
    
    content = ""
    for m in slide_results['results'][0]['matches'][0:3]:
        file = m["metadata"]["file"]
        page = m["metadata"]["Slide"]
        content += f"* [{file}#{page}](https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/lectures/{file}#page={page}) \n"
    result = {}
    result['query'] = query
    result['slides'] = url
    result['markdown'] = markdown.markdown(content)
    logger.bind(user="1").info(f"Topic: {query} |")
    return Response(content=json.dumps(result), media_type="application/json")

@app.get("/question")
async def question(query: str = None):
    index_id = "295-youtube-index"
    pinecone.init(
                api_key=pinecone_api_key,
                environment="us-west1-gcp-free"
                )
    pinecone_index = pinecone.Index(index_id)
    pinecone_index.describe_index_stats()
            
    # Encode the query using the 'text-embedding-ada-002' model
    #encoded_query = client.embeddings.create(input=query,model="text-embedding-ada-002")['data'][0]['embedding']
    encoded_query = client.embeddings.create(input = [query], model="text-embedding-ada-002").data[0].embedding
    response = pinecone_index.query(encoded_query, top_k=20,
                            include_metadata=True)
    context = ""
    for m in response['matches']:
        context += "\n" + m['metadata']['text']
            
    url = ""
    st.subheader("References")
    for m in response['matches'][0:2]:
        url += m['metadata']['url']   
    prompt=f"Please provide a concise answer in markdown format tothe following question: {query} based on content below{context} and the internet. If you are not sure, then say I donot know."
    query_gpt = [
        {"role": "system", "content": "You are a helpful teachingassistant for computer organization"},
        {"role": "user", "content": f"""{prompt}"""},
        ]
    answer_response = client.chat.completions.create(model='gpt-3.5-turbo',
        messages= query_gpt,
        temperature=0,
        max_tokens=500)
    answer = answer_response.choices[0].message.content
    
    
    # Create json from list
    url = []
    for m in response['matches']:
        url.append(m['metadata']['url'])
    logger.bind(user="1").info(f"Topic: {query} |")
    
    response_json = {"answer": answer, "references": url}
    
    return Response(content=json.dumps(response_json), media_type="application/json")



if __name__ == "__main__":
    repo = os.getenv("REPO")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=42000,
        log_level="debug",
        ssl_keyfile=f"{repo}/key.pem",  # Path to your key file
        ssl_certfile=f"{repo}/cert.pem"  # Path to your certificate file
    )
