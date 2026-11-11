import torch 
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi

from transformers import pipeline

pipe = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0,
    dtype=torch.float16,
)


def get_youtube_transcript(url: str) -> str:
    if "v=" in url:
        video_id = url.split("v=", 1)[1].split("&", 1)[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/", 1)[1].split("?", 1)[0]
    else:
        video_id = url

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["hi", "en"])
    return " ".join(snippet.text for snippet in transcript)


def summary(video_url: str) -> str:
    transcript = get_youtube_transcript(video_url)
    output = pipe(transcript, truncation=True)
    return output[0]["summary_text"]


demo = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(
        label="YouTube video URL",
        placeholder="Paste a YouTube video link",
    ),
    outputs=gr.Textbox(label="Summary", lines=8),
    title="YouTube Video Summarizer",
    description="Get a concise summary from a YouTube video's transcript.",
)