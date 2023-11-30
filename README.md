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
export OPENAI_API_KEY=coursekey
export PINECONE_API_KEY=coursekey
docker run -e PINECONE_API_KEY -e OPENAI_API_KEY -p 40000:40000 -it ashriram/askmy295prof streamlit run /repo/app/01_Ask.py
# Navigate to browser 
https://localhost:40000
# To run rest api
docker run -e PINECONE_API_KEY -e OPENAI_API_KEY -it ashriram/askmy295prof python3 /repo/app/fast_api.py
# Run curl requests from command line or paste the https in a browser
curl -k "https://0.0.0.0:42000/topic?query=R-type%0A" # Topic endpoint                             

curl -k "https://0.0.0.0:42000/question?query=What%20is%20a%20R-type%20Instruction" # Question endpoint
```


# Demo video
![Ask My 295 Prof]((./AskMyProf.png))
![Ask My 295 Prof]
(https://www.youtube.com/watch?v=MKRYzsw3t6E)