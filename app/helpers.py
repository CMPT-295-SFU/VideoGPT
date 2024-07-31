def display_slide_results(slide_results):
    content = ""
    for m in slide_results["matches"][0:4]:
        url = m["metadata"]["file"]
        page = m["metadata"]["Slide"]
        content += f"[[{url}#{page}](https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/lectures/{url}#page={page})] "
    return content
        
def get_slide_content(index, encoded_query):
    print(encoded_query)
    slide_results = index.query(
        vector=[encoded_query], top_k=5, namespace="Slides", include_metadata=True
    )
    content = ""
    print(slide_results)
    for r in slide_results["matches"]:
        content += r["metadata"]["description"] + "\n"
    return slide_results,content

def extract_week(slide_results):
    week = (
        slide_results["matches"][0]["metadata"]["file"]
        .split("/")[0]
        .replace("Part", "")
    )
    return week

def display_audio_results(audio_results):
    content = ""
    for m in audio_results["matches"][0:4]:
        url = m["metadata"]["url"]
        content += f"[[{url}]({url})] "
    return content    



def get_audio_content(index, encoded_query):
    audio_results = index.query(
        vector=[encoded_query], top_k=5, include_metadata=True
    )
    content = ""
    for r in audio_results["matches"]:
        content += r["metadata"]["text"] + "\n"

    return audio_results,content
