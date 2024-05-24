import praw
from transformers import pipeline

reddit = praw.Reddit(client_id="0J7GpdceeSnKD4lqLFQJZg",
                        client_secret="lpCPg0PP822S5AXC5Fzb0V8dnbqQ3Q",
                        user_agent="pashtetttt")

# Function to fetch Reddit comments
def fetch_reddit_comments(submission_id, type_of_model):
    submission = reddit.submission(id=submission_id)
    submission.comments.replace_more(limit=None)
    comments = []

    def extract_comments(comment, parent_id=None):

        if type_of_model == "simple_model":
            classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            comment_class = classifier([comment.body])
            comment_data = {
                "id": comment.id,
                "body": comment.body,
                "class": comment_class[0]['label'],
                "parent_id": parent_id,
                "replies": []
            }
            comments.append(comment_data)
            for reply in comment.replies:
                extract_comments(reply, comment.id)
        elif type_of_model == "cyberbullying":
            pass
        else:
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
