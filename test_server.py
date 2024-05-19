import requests
import json
url = f"http://127.0.0.1:8000/get_graph"
response = requests.get(url)
print(response.json())