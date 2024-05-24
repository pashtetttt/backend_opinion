import requests
import pickle
import joblib
import xgboost as xgb
from transformers import BertTokenizer, BertForSequenceClassification
from exceptions import VKUrlFormatException
import re


def get_comments(owner_id, post_id):
    url = f"https://api.vk.com/method/wall.getComments?owner_id={owner_id}&post_id={post_id}&access_token=91e98b2191e98b2191e98b21b692fe59c3991e991e98b21f7e36a817f5095b9dee66625&v=5.199&need_likes=1&count=100"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data['response']
        
    else:
        print("Error: ", response.status_code)
        return None

def get_user_by_id(user_id):
    url = f"https://api.vk.com/method/users.get?user_ids={user_id}&access_token=91e98b2191e98b2191e98b21b692fe59c3991e991e98b21f7e36a817f5095b9dee66625&v=5.199"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return  data['response'][0]["last_name"] + " " + data['response'][0]["first_name"]
        
    else:
        print("Error: ", response.status_code)
        return None    
    
def take_info_for_graph(json_response, type_of_model):
    if type_of_model == "russian_news":
        items = json_response["items"]
        class_user_text = []
        loaded_model = joblib.load("text_classification_model.pkl")
        for i in range(99):
            try:
                user_id = items[i]["from_id"]
                user_name = get_user_by_id(user_id=user_id)
                comment = items[i]["text"]
                text_class = loaded_model.predict([comment])
                properties = {
                "id": user_name,
                "comment": comment,
                "class": text_class[0]
                }
                class_user_text.append(properties)
            except:
                continue
        
    if type_of_model == "lenta_news":
        items = json_response["items"]
        class_user_text = []
        model = xgb.Booster()
        model.load_model("xgboost_model_10_iterations.model")
        with open('count_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        for i in range(99):
            try:
                user_id = items[i]["from_id"]
                user_name = get_user_by_id(user_id=user_id)
                comment = items[i]["text"]
                comment_class = [comment]
                X_new_counts = vectorizer.transform(comment_class)
                dnew = xgb.DMatrix(X_new_counts)
                predictions = model.predict(dnew)
                predicted_labels = label_encoder.inverse_transform(predictions.astype(int))
                label = predicted_labels[0]
                properties = {
                "id": user_name,
                "comment": comment,
                "class": label
                }
                class_user_text.append(properties)
            except:
                print(f"error: {label}")
                continue
    if type_of_model == "toxicity":
        class_user_text = create_graph_toxicity(response=json_response)
    return class_user_text

def define_model(model_type):
    if model_type == "russian_news":
        return "text_classification_model.pkl"
    
def owner_and_post_ids_from_url(url):
    try:
        
        if not url.startswith("https://vk.com/"):
            raise VKUrlFormatException("Неправильный формат ссылки на публикацию Вконтакте")
        
        
        pattern = r'wall(-?\d+)_(\d+)'
        
        # Поиск соответствий в URL
        match = re.search(pattern, url)
        
        if match:
            group_id = match.group(1)
            post_id = match.group(2)
            return group_id, post_id
        else:
            raise VKUrlFormatException("Неправильный формат ссылки на публикацию Вконтакте")
    
    except VKUrlFormatException as e:
        print(e)
        return None, None

def take_info_for_graph_without_model(json_response):
    count = json_response["count"]
    items = json_response["items"]
    user_comment = {}
    for i in range(99):
        try:
            user_id = items[i]["from_id"]
            user_name = get_user_by_id(user_id)
            comment = items[i]["text"]
            user_comment[user_name] = comment
        except:
            continue

    return user_comment

def create_graph_toxicity(response):
    tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
    toxic_model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
    class_user_text = []
    user_comment = take_info_for_graph_without_model(response)
    for key, value in user_comment.items():
        batch = tokenizer.encode(value, return_tensors='pt')
        if toxic_model(batch)['logits'][0][0] > 0:
            pr_class = 'positive'
        else:
            pr_class = 'negative'
        properties = {
            "id": key,
            "comment": value,
            "class": pr_class[0]
        }
        class_user_text.append(properties)
    return class_user_text

