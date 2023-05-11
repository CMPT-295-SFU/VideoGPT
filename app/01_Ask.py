import streamlit as st
from components.sidebar import sidebar
from utils import query_gpt, query_gpt_memory, show_pdf, pinecone_api_key
import pinecone
import openai
import pandas as pd

st.set_page_config(
    page_title="295VideoGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "295VideoGPT is a semantic bot that answers questions about your video lectures",
    },
)

# Session states
# --------------
if "chosen_class" not in st.session_state:
    st.session_state.chosen_class = "--"

if "chosen_pdf" not in st.session_state:
    st.session_state.chosen_pdf = "--"

if "memory" not in st.session_state:
    st.session_state.memory = ""


#sidebar()

st.header("295-VideoGPT") 
st.subheader("Search the topics in 295's videos")
# bucket_name = "classgpt"
# s3 = S3(bucket_name)

# all_classes = s3.list_files()
# all_classes = ["01-pdf","02-pdf"]

# chosen_class = st.selectbox(
#     "Select a class", all_classes + ["--"], index=len(all_classes))

# st.session_state.chosen_class = chosen_class

# if st.session_state.chosen_class != "--":
#     all_pdfs = ["01-pdf", "02-pdf"]
#     chosen_pdf = st.selectbox(
#         "Select a PDF file", all_pdfs + ["--"], index=len(all_pdfs)
#     )

#     st.session_state.chosen_pdf = chosen_pdf

#     if st.session_state.chosen_pdf != "--":
col1, col2 = st.columns(2)

with col1:
#    st.subheader("Lookup a topic")
    st.markdown(
        """
        Here are some examples
        - `R-type instruction`, `Locality`, `Memory allocation`
        """
    )
    query = st.text_area("Enter your topic of interest", max_chars=25)

    if st.button("Ask"):
        if query == "":
            st.error("Please enter a topic")
        with st.spinner("Generating answer..."):
            # res = query_gpt_memory(chosen_class, chosen_pdf, query)
             index_id = "295-youtube-index"

        pinecone.init(
                    api_key=pinecone_api_key,
                    environment="us-west1-gcp-free"
                    )
        #openai.api_key = "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"
        pinecone_index = pinecone.Index(index_id)
        encoded_query = openai.Embedding.create(input=query,model="text-embedding-ada-002")['data'][0]['embedding']
                    #res = query_gpt(chosen_class, chosen_pdf, query)
        response = pinecone_index.query(encoded_query, top_k=3,
                  include_metadata=True)
        elements = []
        st.header("Top 3 Hits")
        for m in response['matches']:
            url = m['metadata']['url']
            st.markdown(f"[{url}]({url})")
            # elements.append({'url': m['metadata']['url']})

        # st.markdown(pd_elements.to_html(escape=False, index=False),unsafe_allow_html=True)
                    # with st.expander("Memory"):
                    #      st.write(st.session_state.memory.replace("\n", "\n\n"))

        # with col2:
        #     show_pdf(chosen_class, chosen_pdf)
