import re
import pandas as pd
import json
from tqdm import tqdm
from src.utils.lm_modeling import load_model, load_text2embedding
import torch
from torch_geometric.data.data import Data

model_name = 'sbert'
edge_graph = pd.read_csv("edges.tsv", sep='\t')

def textualize_graph(edge_graph, node_graph):
    tuplets = re.findall(r'\((.*?)\)', edge_graph)
    nodes = {} #dict
    edges = [] #list
    for tup in tuplets:
        src, dst, freq, avg = tup.split(';')
        src = src.strip()
        dst = dst.strip()
        avg = round(float(avg))
        if src not in nodes:
            nodes[src] = len(nodes) # id of node
        if dst not in nodes:
            nodes[dst] = len(nodes) #id of node
        edges.append({'src': nodes[src], 'freq': freq, 'avg_time': avg, 'dst': nodes[dst], })

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)

    # Merge the nodes DataFrame with the JSON DataFrame
    detailed_nodes_df = pd.merge(nodes, node_graph, on='node_attr', how='left')

    #print(detailed_nodes_df)
    #print(edges)
    return detailed_nodes_df, edges

def step_one():
     
    with open("nodes.json", 'r') as f:
        node_graph = json.load(f)
    
    nodes_df = pd.DataFrame.from_dict(node_graph, orient='index').reset_index()
    nodes_df.columns = ['node_attr'] + list(nodes_df.columns)[1:]

    nodes, edges = textualize_graph(edge_graph['graph'][0], nodes_df)
    
    nodes.to_csv("nodes_textualized.csv", index=False, columns= nodes.columns)
    edges.to_csv('edges_textualized.csv', index=False, columns=['src', 'freq', 'avg_time', 'dst'])

def step_two():

    def encode_graph():
        print('Encoding graphs...')
        nodes = pd.read_csv("nodes_textualized.csv")
        edges = pd.read_csv("edges_textualized.csv")
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.src, edges.dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
        torch.save(data,'graphs.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    encode_graph()


if __name__ == '__main__':
    step_one()
    step_two()