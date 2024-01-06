# %%

from __future__ import unicode_literals

import sys

if __package__ is None and not hasattr(sys, 'frozen'):
    # direct call of __main__.py
    import os.path
    path = os.path.realpath(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))


import base64
import collections
import glob
import json
import optparse
import os
import pickle
import sys
import time
import webbrowser
from io import open
from itertools import islice

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import httplib2
import langchain
import numpy as np
import oauth2client
import openai
import pinecone
import requests
import tiktoken
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import HumanMessage, SystemMessage
from oauth2client import file
from openai import OpenAI
from tenacity import (retry, retry_if_not_exception_type, stop_after_attempt,
                      wait_random_exponential)
from tokenizers import Tokenizer
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
import argparse


def create_chapters_from_transcript(transcript_json, chapter_length_sec=300):
    """
    Creates chapters from a YouTube video transcript.

    :param transcript_json: JSON string containing the transcript and timestamps.
    :param chapter_length_sec: Length of each chapter in seconds (default 300 seconds).
    :return: JSON string containing the chapters.
    """
    try:
        # Load transcript data from JSON
        transcript_data = json.loads(transcript_json)

        chapters = []
        current_chapter = {"start_time": 0, "end_time": 0, "texts": []}
        for entry in transcript_data:
            if entry['start'] >= current_chapter["end_time"]:
                if current_chapter["texts"]:
                    chapters.append(current_chapter)
                current_chapter = {
                    #   "start_time_ms": start_time, "end_time_ms": end_time,
                    "start_time": entry['start'], "end_time": entry['start'] + chapter_length_sec, "texts": []}
            current_chapter["texts"].append(entry['text'])

        # Add the last chapter
        if current_chapter["texts"]:
            chapters.append(current_chapter)

        return json.dumps(chapters, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_transcript(video_id):
    """
    Retrieves the transcript of a YouTube video along with timestamps.

    :param video_id: The ID of the YouTube video.
    :return: JSON string containing the transcript and timestamps.
    """
    try:
        # Retrieve the available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Fetch the transcript in the desired language
        transcript = transcript_list.find_transcript(
            ['en'])  # 'en' for English

        # Fetch the actual transcript data
        transcript_data = transcript.fetch()

        # Format the transcript data into a JSON structure
        transcript_json = json.dumps(transcript_data, indent=4)
        return transcript_json
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_youtube_handler(client_secrets_file, request_token_file):
    """
    Creates a YouTube API handler using client secrets and request token.

    :param client_secrets_file: Path to the client_secrets.json file.
    :param request_token_file: Path to the request.token file.
    :return: Authenticated YouTube API client.
    """
    # Load client secrets
    with open(client_secrets_file) as f:
        client_info = json.load(f)['web']

    # Load request token
    with open(request_token_file) as f:
        request_token = json.load(f)

    # Add client_id and client_secret to request_token
    request_token['client_id'] = client_info['client_id']
    request_token['client_secret'] = client_info['client_secret']

    # Create credentials from request_token
    credentials = Credentials.from_authorized_user_info(request_token)

    # Refresh the credentials if necessary
    if credentials.expired:
        try:
            credentials.refresh(Request())
        except:
            # Load client secrets
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes)

            # Run the flow to get the credentials
            credentials = flow.run_local_server(open_browser=True)

    # Build the YouTube API client
    youtube = build('youtube', 'v3', credentials=credentials)

    # Save the credentials for future use
    with open('token.pickle', 'wb') as token:
        pickle.dump(credentials, token)

    # Build the YouTube API client
    youtube = build('youtube', 'v3', credentials=credentials)
    return youtube


# import auth

channelId = "UC6TMMgA1zibQxNxvT7qtnpw"
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def get_transcript_for_playlist():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "./client_secrets.json"

    youtube = get_youtube_handler("./client_secrets.json", "./request.token")
    request = youtube.playlists().list(
        part="snippet,contentDetails",
        channelId=channelId,
        maxResults=50  # Adjust the maxResults as needed
    )

    response = request.execute()

    # print(response)

    for playlist in response["items"]:
        print(playlist["snippet"]["title"]+":")
        playlist_name = playlist["snippet"]["title"]
        # Create folder for playlist
        if not os.path.exists("transcripts/"+playlist_name):
            os.makedirs("transcripts/"+playlist_name)
        root = "transcripts/"+playlist_name
        playlist_request = youtube.playlistItems().list(
            playlistId=playlist["id"],
            part="snippet"
        )
        playlist_response = playlist_request.execute()
        for video in playlist_response["items"]:
            print("  - video: \"" + video["snippet"]["title"] + "\"")
            print("    url: \"" + video["snippet"]
                  ["resourceId"]["videoId"] + "\"")
            print("    youtube: 1")
            # Get the transcript and write it to a file
            transcript_json = get_transcript(
                video["snippet"]["resourceId"]["videoId"])
            with open(f"{root}/{video['snippet']['resourceId']['videoId']}.txt", "w") as f:
                f.write(transcript_json)


def process_videos(video):
    api_key = "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"
    client = OpenAI(api_key=api_key)
    chat = ChatOpenAI(model="gpt-4-1106-preview",  # "gpt-4-vision-preview",
                      max_tokens=4096, openai_api_key=api_key)

    with open(f"{video}", "r") as f:
        text = f.read()
    text = create_chapters_from_transcript(text)
    video_title = os.path.basename(video).split(".")[0]
    print(video, video_title)
    content_json = chat.invoke(
        [
            SystemMessage(content="You are an algorithm tasked with analyzing a video transcript. The transcript is a JSON object with each entry with the following: text (audio closed captions), start (timestamp), duration (seconds of the text). Concisely summarize and create less than 10 to 15 chapters. Return a json with the following fields: title (a few words summarizing the text), start_time (float))"),
            HumanMessage(
                content=[
                    {"type": "text",
                        "text": text}
                ]
            )
        ])

    # # print(content_json.content)
    content_string = content_json.content.strip('```json\n')
    # print(content_string)
    # print(content_string)
    # Get all text etween [ and ]
    content_string = content_string[content_string.find(
        "["):content_string.rfind("]")+1]
    # print(content_string)
    content_dict = json.loads(content_string)

    for i, entry in enumerate(content_dict):
        # Check if first iteration of loop
        if i == 0:
            entry["start_time"] = "00:00"
        else:
            # if start_time exceeds hour mark, convert to hours
            if entry["start_time"] >= 3600:
                entry["start_time"] = time.strftime(
                    '%H:%M:%S', time.gmtime(entry['start_time']))
            else:
                entry["start_time"] = time.strftime(
                    '%M:%S', time.gmtime(entry['start_time']))

    # # Iterate over json and combine title and start_time into description field for youtube video
    description = ""
    for entry in content_dict:
        description += f"{entry['start_time']} - {entry['title']}\n"

    # # Get youtube handler and update description of video
    # print(description)
    youtube = get_youtube_handler("./client_secrets.json", "./request.token")
    # Create request to get video's title
    request = youtube.videos().list(
        part="snippet",
        id=video_title
    )
    video_data = request.execute()
    title = video_data["items"][0]["snippet"]["title"]
    categoryId = video_data["items"][0]["snippet"]["categoryId"]
    # print(title)
    request = youtube.videos().update(
        part="snippet",
        body={"id": video_title, "snippet": {"title": title,                "description": description, "categoryId": categoryId}})
    request.execute()

    # # Render as Markdown
    # from IPython.display import Markdown
    # display(Markdown(content_json.content))


def process_playlist(playlist_folder):
    # Iterate over all files in the playlist folder
    files = glob.glob(f"{playlist_folder}/*.txt")
    for v in files:
        process_videos(v)


def main():
    parser = argparse.ArgumentParser(
        description='Process YouTube videos and playlists.')
    parser.add_argument('-u', '--update', type=str,
                        help='Path to the playlist folder to process')
    parser.add_argument('-d', '--download', type=str,
                        help='URL of the playlist to download')
    parser.add_argument('-t', '--transcript', type=str,
                        help='Video ID to download transcript')
    parser.add_argument('-v', '--video', type=str,
                        help='Video file to process')

    args = parser.parse_args()

    if args.update:
        process_playlist(args.update)
    elif args.download:
        get_transcript_for_playlist()
    elif args.transcript:
        print(get_transcript(args.transcript))
    elif args.video:
        process_videos(args.video)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # process_playlist("transcripts/Venus")
    # # with open("transcripts/Week10/2-wadCV79F4.txt", "w") as f:
    # #    f.write(get_transcript("2-wadCV79F4"))
    # # process_videos("transcripts/Week10/2-wadCV79F4.txt")
