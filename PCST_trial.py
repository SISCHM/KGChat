import re
import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt


#  generate query embedding using SBERT

def get_query_embedding(query, repeat_factor, tokenizer, model):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1)
    query_emb = query_emb.repeat(1, repeat_factor)  # Repeat the embedding to match the graph dimension
    return query_emb


# Modified PCST with both query embeddings (q_emb_nodes, q_emb_edges)

def retrieval_via_pcst(graph, q_emb_nodes, q_emb_edges, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'frequency', 'avg_time', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.functional.cosine_similarity(q_emb_nodes, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.functional.cosine_similarity(q_emb_edges, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'frequency', 'avg_time', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc


def visualize_subgraph(subgraph, textual_nodes, textual_edges):
    G_sub = nx.DiGraph()
    selected_nodes = subgraph.edge_index.unique().tolist()
    
    for i in selected_nodes:
        node_key = textual_nodes.iloc[i]['node_attr']
        G_sub.add_node(i, label=node_key, average_cost=textual_nodes.iloc[i]['average_cost'], average_invoiced_price=textual_nodes.iloc[i]['average_invoiced_price'])

    for i, (src, dst) in enumerate(subgraph.edge_index.t().tolist()):
        freq = subgraph.edge_attr[i][0].item()
        avg_time = subgraph.edge_attr[i][1].item()
        print(f"Adding edge from {src} to {dst} with frequency {freq} and average time {avg_time}")
        G_sub.add_edge(src, dst, frequency=freq, avg_time=avg_time)

    pos = nx.spring_layout(G_sub)
    labels = nx.get_node_attributes(G_sub, 'label')
    edge_labels = {(src, dst): f'Freq: {data["frequency"]}\nAvg Time: {data["avg_time"]}' for src, dst, data in G_sub.edges(data=True)}

    nx.draw(G_sub, pos, labels=labels, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels, font_size=6)
    plt.show()



def main():

    graph = torch.load('graphs.pt')

    model_name = 'sentence-transformers/all-roberta-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    query = "What comes after Payment?" #change and evaluate result
    q_emb_nodes = get_query_embedding(query, 3, tokenizer, model)  # for nodes
    q_emb_edges = get_query_embedding(query, 2, tokenizer, model)  # for edges


    textual_nodes = pd.read_csv("nodes_textualized.csv")
    textual_edges = pd.read_csv("edges_textualized.csv")
    textual_edges = textual_edges.rename(columns={'src': 'src', 'dst': 'dst', 'freq': 'frequency', 'avg_time': 'avg_time'})

    # retrieval
    subgraph, description = retrieval_via_pcst(graph, q_emb_nodes, q_emb_edges, textual_nodes, textual_edges)

    # visualize 
    visualize_subgraph(subgraph, textual_nodes, textual_edges)


if __name__ == "__main__":
    main()
