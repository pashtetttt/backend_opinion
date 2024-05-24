import praw
from transformers import pipeline
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder

reddit = praw.Reddit(client_id="0J7GpdceeSnKD4lqLFQJZg",
                        client_secret="lpCPg0PP822S5AXC5Fzb0V8dnbqQ3Q",
                        user_agent="pashtetttt")

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
            df = pd.read_csv('./datasets/cyberbullying_tweets.csv')
            label_encoder = LabelEncoder()
            label_encoder.fit(df['cyberbullying_type'])
            tokenizer = DistilBertTokenizer.from_pretrained('./cyberbullying_tokenizer')
            model = DistilBertForSequenceClassification.from_pretrained('./cyberbullying_model')
            inputs = tokenizer([comment.body], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            predicted_label = label_encoder.inverse_transform(predictions.numpy())
            comment_data = {
                "id": comment.id,
                "body": comment.body,
                "class": predicted_label[0],
                "parent_id": parent_id,
                "replies": []
            }
            comments.append(comment_data)
            for reply in comment.replies:
                extract_comments(reply, comment.id)
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
