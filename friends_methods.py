import requests
import json
def get_group_members(access_token, group_id):
    url = "https://api.vk.com/method/groups.getMembers"
    params = {
        "group_id": group_id,
        "access_token": access_token,
        "v": "5.131"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data:
        raise Exception(data["error"]["error_msg"])
    return data["response"]["items"]

def get_user_friends(access_token, user_id):
    url = "https://api.vk.com/method/friends.get"
    params = {
        "user_id": user_id,
        "access_token": access_token,
        "v": "5.131",
        "fields": "nickname"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data:
        raise Exception(data["error"]["error_msg"])
    return data["response"]["items"]

def get_user_info(access_token, user_ids):
    url = "https://api.vk.com/method/users.get"
    params = {
        "user_ids": ",".join(map(str, user_ids)),
        "access_token": access_token,
        "v": "5.131",
        "fields": "nickname"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "error" in data:
        raise Exception(data["error"]["error_msg"])
    return data["response"]

def get_group_user_connections(access_token, group_id):
    try:
        group_members = get_group_members(access_token, group_id)
        user_info = get_user_info(access_token, group_members)

        result = []
        for user in user_info:
            user_id = user["id"]
            username = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
            friends = get_user_friends(access_token, user_id)
            friend_ids = [friend["id"] for friend in friends if friend["id"] in group_members]

            result.append({
                "id": user_id,
                "username": username,
                "friends": friend_ids
            })
        
        return result

    except Exception as e:
        print(f"Error: {e}")
        return []

# Пример использования
access_token = "91e98b2191e98b2191e98b21b692fe59c3991e991e98b21f7e36a817f5095b9dee66625"
group_id = "224867582"  # Например, '177976851' для группы
connections = get_group_user_connections(access_token, group_id)


print(json.dumps(connections, ensure_ascii=False, indent=2))
