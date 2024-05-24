from applicaton_methods import get_comments, get_user_by_id, take_info_for_graph
import requests
import json
import joblib


post_id = 23572450
owner_id = -29534144
response = get_comments(owner_id, post_id)

class_user_text = []
    user_comment = take_info_for_graph(response)
    print(user_comment)
    for key, value in user_comment.items():
        pr_class = loaded_model.predict([value])
        properties = {
            "id": key,
            "comment": value,
            "class": pr_class[0]
        }
        class_user_text.append(properties)


    file_path = "data_for_graph.json"

    with open(file_path, "w") as json_file:
        # Write the JSON data to the file
        json.dump(class_user_text, json_file)

    print("JSON data has been saved to", file_path)