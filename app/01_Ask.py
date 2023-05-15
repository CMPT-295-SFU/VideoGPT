

import streamlit as st
from components.sidebar import sidebar
from utils import query_gpt, query_gpt_memory, show_pdf, pinecone_api_key
import pinecone
import openai
import pandas as pd
from loguru import logger
import sys
from streamlit_star_rating import st_star_rating
from streamlit_text_rating.st_text_rater import st_text_rater



st.set_page_config(
    page_title="295VideoGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "295VideoGPT is a semantic bot that answers questions about your video lectures",
    },
)

if 'count' not in st.session_state:
    st.session_state.count = 0

# Add a rotating log file that rotates every day
#logger.add("logs/myapp_{time:YYYY-MM-DD}.log", rotation="1 week")

# Add a console logger with a custom message format


st.header("295-VideoGPT") 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Topic Search (only top 3 matches provided)")
#    st.subheader("Lookup a topic")
    st.markdown(
        """
        Here are some examples
        - `R-type instruction`, `Locality`, `Memory allocation`
        """
    )
    query = st.text_area("Enter topic you want to find in videos", max_chars=25)

    if st.button("Ask"):
        current = st.session_state.count
        st.session_state.count += 1
        if query == "":
            st.error("Please enter a topic")
        with st.spinner("Generating answer..."):
            # res = query_gpt_memory(chosen_class, chosen_pdf, query)
            index_id = "295-youtube-index"
            pinecone.init(
                    api_key=pinecone_api_key,
                    environment="us-west1-gcp-free"
                    )
            pinecone_index = pinecone.Index(index_id)
            encoded_query = openai.Embedding.create(input=query,    model="text-embedding-ada-002")['data'][0]['embedding']
                        #res = query_gpt(chosen_class, chosen_pdf, query)
            response = pinecone_index.query(encoded_query, top_k=3,
                      include_metadata=True)
            elements = []
            # st.header("Top 3 Hits")
            for m in response['matches']:
                url = m['metadata']['url']
                st.markdown(f"[{url}]({url})")
        logger.bind(user=st.session_state.count).info(f"Topic: {query} |")
 
        # Logging         # elements.append({'url': m['metadata']['url']})

with st.container():
    st.subheader("Question-Answering: I am feeling Lucky")
#    st.subheader("Lookup a topic")
    st.markdown(
        """
        Here are some examples
        - What is a pointer and give me an example ?
        - How to find the address of an element in a 2D array ? 
        """
        )
    query = st.text_area("Question", max_chars=200)
    if st.button("Answer"):
        st.session_state.count += 1
        if query == "":
            st.error("Please enter a topic")
        with st.spinner("Generating answer.....(can take upto 20 seconds)"):
            index_id = "295-youtube-index"
            pinecone.init(
                        api_key=pinecone_api_key,
                        environment="us-west1-gcp-free"
                        )
            pinecone_index = pinecone.Index(index_id)
            pinecone_index.describe_index_stats()
            openai.api_key ="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"
            # Encode the query using the 'text-embedding-ada-002' model
            encoded_query = openai.Embedding.create(input=query,model="text-embedding-ada-002")['data'][0]['embedding']
            response = pinecone_index.query(encoded_query, top_k=20,
                              include_metadata=True)
            context = ""
            for m in response['matches']:
                context += "\n" + m['metadata']['text']
            
            url = ""
            st.subheader("References")
            for m in response['matches'][0:2]:
                url += m['metadata']['url']   
            prompt=f"Please provide a concise answer in markdown format to the following question: {query} based on content below {context} and the internet. If you are not sure, then say I do not know."
            query_gpt = [
            {"role": "system", "content": "You are a helpful teaching assistant for computer organization"},
            {"role": "user", "content": f"""{prompt}"""},
            ]
            answer_response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages= query_gpt,
            temperature=0,
            max_tokens=500,
            )
            st.markdown(answer_response["choices"][0]["message"]["content"])        
            st.markdown("**References**")
            for m in response['matches'][0:2]:
                url = m['metadata']['url']
                st.markdown(f"[{url}]({url})")
            logger.bind(user=st.session_state.count).info(f"Q/A: {query}|")
            




