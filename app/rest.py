import asyncio
import json
import logging
import os
import sys
from typing import AsyncGenerator, AsyncIterable, NoReturn

import markdown
import pandas as pd
import pinecone
import streamlit as st
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Response, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_community.embeddings.openai import OpenAIEmbeddings


from openai import OpenAI

# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI

# from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from helpers import (
    display_audio_results,
    display_slide_results,
    get_audio_content,
    get_slide_content,
    extract_week,
)
from components.sidebar import sidebar

# set to DEBUG for more verbose logging
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
# s3 = S3("classgpt")

# Flag to prevent duplicate logger initialization
_logger_initialized = False

def init_logger():
    global _logger_initialized
    if _logger_initialized:
        return
    
    # Remove default logger to prevent duplicates
    logger.remove()
    
    logger.add(
        "295API.log",
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message} | {extra[user]}",
        rotation="10 MB",
        compression="gz",
    )
    _logger_initialized = True


# Initialize logger
init_logger()

# client = OpenAI(api_key="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")
client = OpenAI(api_key=openai_api_key)
em_client = OpenAIEmbeddings(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("295-vstore")

app = FastAPI()
client = AsyncOpenAI()

# Add startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("Application shutting down...")
    # Close any open connections if needed
    try:
        await client.close()
    except Exception as e:
        logger.error(f"Error closing OpenAI client: {e}")

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


# Print the data posted
@app.post("/rating")
async def rating(request: Request):
    # Print the post body
    data = await request.json()
    rating = data["rating"]
    query = data["query"].replace("\n", "%20")
    logger.bind(user="1").info(f" Query: {query} | Rating: {rating} ")
    return {"message": "Logged"}


@app.get("/topic")
async def topic(query: str = None):
    try:
        print(query)
        text = query.replace("\n", " ")
        # encoded_query = await (
        # client.embeddings.create(input=[text], model="text-embedding-ada-002"))
        # encoded_query = encoded_query.data[0].embedding
        encoded_query = em_client.embed_documents([text])[0]
        context = "Question: " + query + "\n"
        context += "\n" + "#######Slide Context#####\n"
        slide_results, content = get_slide_content(index, encoded_query)
        context += "\n #######Audio Context#####\n"
        audio_results, content = get_audio_content(index, encoded_query)
        context += content
        elements = []
        # Create json from list
        url = []
        print(slide_results)
        for m in slide_results["matches"]:
            url.append(m["metadata"]["file"])

        response = ""
        response += "\n\n #### Slide References \n\n" + display_slide_results(slide_results)
        response += "\n\n #### Audio References \n\n" + display_audio_results(audio_results)

        result = {}
        result["query"] = query
        result["slides"] = url
        result["markdown"] = response
        query = query.replace("\n", "%20")
        logger.bind(user="1").info(f"Topic: {query}")
        return Response(content=result["markdown"], media_type="application/text")
        
    except Exception as e:
        logger.error(f"Error in topic endpoint: {e}")
        error_response = f"Error occurred: {str(e)}"
        return Response(content=error_response, media_type="application/text", status_code=500)


@app.get("/question")
async def question(query: str = None):
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("295-vstore")
        index.describe_index_stats()
        encoded_query = em_client.embed_documents([query])[0]
        
        response = index.query(vectors=[encoded_query], top_k=20, include_metadata=True)
        
        context = ""
        for m in response["matches"]:
            context += "\n" + m["metadata"]["text"]

        url = ""
        for m in response["matches"][0:2]:
            url += m["metadata"]["url"]
            
        prompt = f"Please provide a concise answer in markdown format tothe following question: {query} based on content below{context} and the internet. If you are not sure, then say I donot know."
        query_gpt = [
            {
                "role": "system",
                "content": "You are a helpful teachingassistant for computer organization",
            },
            {"role": "user", "content": f"""{prompt}"""},
        ]
        
        answer_response = client.chat.completions.create(
            model="gpt-4o-mini", messages=query_gpt, temperature=0, max_tokens=500
        )
        answer = answer_response.choices[0].message.content

        # Create json from list
        url = []
        for m in response["matches"]:
            url.append(m["metadata"]["url"])

        response_json = {"answer": answer, "references": url}
        return Response(content=json.dumps(response_json), media_type="application/json")
        
    except Exception as e:
        logger.error(f"Error in question endpoint: {e}")
        error_response = {"error": f"An error occurred: {str(e)}", "references": []}
        return Response(content=json.dumps(error_response), media_type="application/json", status_code=500)


async def get_ai_response(query: str) -> AsyncGenerator[str, None]:
    """
    OpenAI Response with proper error handling
    """
    try:
        encoded_query = await client.embeddings.create(
            input=[query], model="text-embedding-ada-002"
        )
        encoded_query = encoded_query.data[0].embedding

        context = "Question: " + query + "\n"
        context += "\n" + "#######Slide Context#####\n"
        slide_results, content = get_slide_content(index, encoded_query)
        context += content
        context += "\n #######Audio Context#####\n"
        audio_results, content = get_audio_content(index, encoded_query)
        context += content

        # st.markdown(display_audio_results(audio_results))

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in RISC-V, Computer architecture and Computer organization."
                        "Try to answer the following question based on the"
                        "context below in markdown format. If you do not know the answer, just say 'I don't know'."
                    ),
                },
                {
                    "role": "user",
                    "content": context,
                },
            ],
            stream=True,
        )

        all_content = ""
        try:
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    all_content += content
                    yield all_content
        except Exception as stream_error:
            logger.error(f"Error in streaming response: {stream_error}")
            yield f"Error occurred while streaming response: {str(stream_error)}"
            return

        all_content += "\n\n #### Slide References \n\n" + display_slide_results(
            slide_results
        )
        all_content += "\n\n #### Audio References \n\b" + display_audio_results(
            audio_results
        )
        week = extract_week(slide_results)
        query = query.replace("\n", "%20")
        logger.bind(user="1").info(f"Query: {query}")
        all_content += (
            "\n\n #### Relevant Week \n"
            + f"[Relevant Week's videos](https://www.cs.sfu.ca/~ashriram/Courses/CS295/videos.html#week{week})`you can lookup using provided slide reference above`"
        )
        yield all_content
        yield "EOS"
        
    except Exception as e:
        logger.error(f"Error in get_ai_response: {e}")
        yield f"Error occurred: {str(e)}"
        yield "EOS"


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Websocket for AI responses
    """
    await websocket.accept()
    try:
        while True:
            try:
                # Add timeout to prevent hanging connections
                message = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                
                async for text in get_ai_response(message):
                    try:
                        await websocket.send_text(text)
                    except Exception as send_error:
                        logger.error(f"Error sending WebSocket message: {send_error}")
                        break
                        
            except asyncio.TimeoutError:
                logger.warning("WebSocket timeout - closing connection")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message handling: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as close_error:
            logger.error(f"Error closing WebSocket: {close_error}")


if __name__ == "__main__":
    repo = os.getenv("REPO")
    uvicorn.run(
        "rest:app",
        host="0.0.0.0",
        port=42000,
        log_level="info",  # Changed from debug to reduce log noise
        ssl_keyfile=f"{repo}/key.pem",  # Path to your key file
        ssl_certfile=f"{repo}/cert.pem",  # Path to your certificate file
        reload=False,  # Disabled reload for production stability
        workers=1,  # Use single worker to prevent duplicate logs
        access_log=True,  # Enable access logs
        timeout_keep_alive=30,  # Keep alive timeout
        limit_concurrency=100,  # Limit concurrent connections
        limit_max_requests=1000,  # Restart worker after N requests to prevent memory leaks
    )
