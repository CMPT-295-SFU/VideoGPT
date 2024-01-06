
## Slide inputs

- **Step 1: Convert a slide deck in pdf to a folder with one png per slide** 
convert -resize x960 -quality 100 -density 200 -colorspace sRGB "../L00-Intro.pdf" --flatten Slide%02d.png

- **Step 2: Use an LLM to describe the image, convert the description to text and vectorize it. We upload the vectors to pinecone but any vector database can be used**
```bash
# Part11 is the week
# L29 and L30 are folders with one png per slide frame e.g., Slide01.png
$ ls png/Part11/
Part11/L29-Hazard/*.png Part11/L30-Pipeline/*.png
# Get an LLM to describe the image in a verbose manner
# Upload it to pinecone
$ python3 -m fire image_input.py weeklypng_to_vectors --path="png/Part11/*/
```

## Youtube Audio inputs

`Designed to run on google collab or machine with gpu support`

- **Step 1: Download the youtube video as mp3**
- **Step 2: Use google collab to run whisper and transcribe the audio to text**



## Summarize chapters in each audio transcript

- **Download the transcripts for each video in playlist**
  ```bash
  python3 youtube-list.py -d [Youbue playlist id]
  ```
- **Download the transcript**
  ```bash
  python3 youtube-list.py -t [Video ID to download transcript]
  ```
- **Update all videos in playlist**
  ```bash
  python3 youtube-list.py -u [Folder with transcripts]
  ```
- **Update a single video**
  ```bash
  python3 youtube-list.py -v [Video file to process]
  ```
