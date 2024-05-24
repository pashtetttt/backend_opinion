import networkx as nx
import matplotlib.pyplot as plt
import json
import d3graph
# Specify the file path from which you want to load the JSON data
file_path_svm = "data_for_graph.json"
file_path_toxic = "toxic_data.json"
# Open the file in read mode
with open(file_path_svm, "r") as json_file:
    # Load the JSON data from the file
    data = json.load(json_file)

print(data[1])

G = nx.Graph()


for i in range(len(data)):
    vert = data[i]
    for j in range(i+1, len(data)):
        to_connect = data[j]
        if vert['class'] == to_connect['class']:
            G.add_edge(vert['id'], to_connect['id'])

matrix = nx.adjacency_matrix(G)
print(matrix)
d3 = d3graph.d3graph()
d3.graph(matrix)
d3.show()
# nx.draw_circular(G, with_labels=True)
# plt.show()
# print("drew")