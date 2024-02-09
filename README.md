# Video GPT

```

streamlit run app/01_Ask.py
```

> ChatGPT for my lecture slides

# Dockerfile

```
# Building from scratch
docker build -t ashriram/askmy295prof .
# Pull
docker pull ashriram/askmy295prof

# To run. Set environment variables
# Get openai and pinecone keys from the instructor
export OPENAI_API_KEY=coursekey
export PINECONE_API_KEY=coursekey
export REPO=/repo/
docker run -e PINECONE_API_KEY -e OPENAI_API_KEY -p 40000:40000 -it ashriram/askmy295prof streamlit run /repo/app/01_Ask.py
# Navigate to browser 
https://localhost:40000
# To run rest api
docker run -e PINECONE_API_KEY -e OPENAI_API_KEY -e REPO -it ashriram/askmy295prof python3 /repo/app/rest.py
# Open react.html in browser
```


# Demo video
https://www.youtube.com/watch?v=MKRYzsw3t6E
![Ask My 295 Prof](./AskMyProf.png)