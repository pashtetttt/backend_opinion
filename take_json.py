from reddit_data import fetch_reddit_comments
import json
file_path = 'reddit_data.json'


comments = fetch_reddit_comments("1czfqo5", "simple_model")

with open(file_path, "w") as json_file:
    json.dump(comments, json_file)

    print("data saved")