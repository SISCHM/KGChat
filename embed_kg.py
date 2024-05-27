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
    quadruplets = re.findall(r'\((.*?)\)', edge_graph)
    nodes = {} #dict
    edges = [] #list
    for quad in quadruplets:
        src, dst, freq, avg = quad.split(';')
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

    return detailed_nodes_df, edges

def textualize():
     
     # load the data (dict) from the json file describing the nodes of the KG
    with open("nodes.json", 'r') as f:
        node_graph = json.load(f)
    
    # convert the data to a dataframe
    nodes_df = pd.DataFrame.from_dict(node_graph, orient='index').reset_index()
    # insert a column to the dataframe that contains the different activites 
    nodes_df.columns = ['node_attr'] + list(nodes_df.columns)[1:]

    nodes, edges = textualize_graph(edge_graph['graph'][0], nodes_df)
    
    # create csv files that contained the textualized nodes and edges for embedding
    nodes.to_csv("nodes_textualized.csv", index=False, columns= nodes.columns)
    edges.to_csv('edges_textualized.csv', index=False, columns=['src', 'freq', 'avg_time', 'dst'])

def embed():

    def encode_graph():
        print('Encoding graphs...')
        nodes = pd.read_csv("nodes_textualized.csv")
        edges = pd.read_csv("edges_textualized.csv")

        # embedding the different node attributes seperately 
        nodes_attr_emb = text2embedding(model, tokenizer, device, nodes["node_attr"].tolist())
        nodes_avg_cost_emb = text2embedding(model, tokenizer, device, nodes["average_cost"].astype(str).tolist())
        nodes_avg_inv_emb = text2embedding(model, tokenizer, device, nodes["average_invoiced_price"].astype(str).tolist())

        # concatenate / combine the node embeddings
        x = torch.cat([nodes_attr_emb, nodes_avg_cost_emb, nodes_avg_inv_emb], dim=1)

        # embedding the different edge attributes seperately
        edges_freq_emb = text2embedding(model, tokenizer, device, edges["freq"].astype(str).tolist())
        edges_avg_time_emb = text2embedding(model, tokenizer, device, edges["avg_time"].astype(str).tolist())
        
        # concatenate / combine the edge embeddings
        e = torch.cat([edges_freq_emb, edges_avg_time_emb], dim=1)

        # create the edge index tensor 
        edge_index = torch.LongTensor([edges.src, edges.dst])

        # create a data object that represents the knowledge graph
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

        # save the data object to a file 
        torch.save(data,'graphs.pt')

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    encode_graph()


if __name__ == '__main__':
    textualize()
    embed()