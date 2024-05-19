from fastapi import FastAPI, Response
from PIL import Image
import io
import json
from fastapi.middleware.cors import CORSMiddleware
from applicaton_methods import owner_and_post_ids_from_url, get_user_by_id, get_comments, take_info_for_graph

origins = [
    "http://localhost:8080",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/get_graph")
# async def get_graph():
#     # with open("toxic_data.json", "r") as file:
#     #     toxic_data = file.read()

#     # return Response(content=toxic_data)
#     pass

@app.get("/get_graph/{owner_id}/{post_id}/{type_of_model}")
async def get_graph_by_url(owner_id: int, post_id: int, type_of_model: str):
    response = get_comments(owner_id=owner_id, post_id=post_id)

    user_comment_class = take_info_for_graph(json_response=response, type_of_model=type_of_model)
    rs = json.dumps(user_comment_class)
    return Response(content=rs)
    # get ip of post
    # get data from post
    ##### create json
    # deliver content   