import praw
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)

# Function to fetch Reddit comments
def fetch_reddit_comments(submission_id):
    submission = reddit.submission(id=submission_id)
    submission.comments.replace_more(limit=None)
    comments = []

    def extract_comments(comment, parent_id=None):
        comment_data = {
            "id": comment.id,
            "body": comment.body,
            "parent_id": parent_id,
            "replies": []
        }
        comments.append(comment_data)
        for reply in comment.replies:
            extract_comments(reply, comment.id)

    for top_level_comment in submission.comments:
        extract_comments(top_level_comment)
    
    return comments
