import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from pcst_fast import pcst_fast
from torch_geometric.data import Data

class Graph:
    def __init__(self, nodes=None, edges=None, description=None, graph=None):
        self.nodes = nodes
        self.edges = edges
        self.description = description
        self.graph = graph

    def set_edges(self, edges):
        self.edges = edges

    def set_nodes(self, nodes):
        self.nodes = nodes

    def split_label(self, text, max_line_length=10):
        words = text.split(' ')
        lines = []
        current_line = words[0]
        for word in words[1:]:
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return '\n'.join(lines)

    def visualize_graph(self):
        g = nx.DiGraph()
        node_attributes = self.nodes.columns.tolist()
        for i, node in self.nodes.iterrows():
            node_attr_dict = {attr: node[attr] for attr in node_attributes}
            node_attr_dict['label'] = self.split_label(node['node_name'])
            g.add_node(i, **node_attr_dict)

        for i, row in self.edges.iterrows():
            g.add_edge(row['Source_id'], row['Destination_id'], Frequency=row['Frequency'], Average_time=row['Average_time'])

        pos = nx.spring_layout(g, iterations=200, k=5)

        # adjust node size based on the label
        node_labels = nx.get_node_attributes(g, 'label')
        node_sizes = {node: len(label) * 200 for node, label in node_labels.items()}  # Play with the factor 200

        # adjust edge positions
        ax = plt.gca()
        for edge in g.edges(data=True):
            src, dst = edge[0], edge[1]
            arrowprops = dict(arrowstyle='-|>', color='grey')
            ax.annotate('', xy=pos[dst], xytext=pos[src], arrowprops=arrowprops)

        nx.draw_networkx_nodes(g, pos, node_size=[node_sizes[node] for node in g.nodes()], node_color="lightblue", edgecolors='k', linewidths=1)
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=10, font_color='black', font_weight='bold')

        edge_labels = {(src, dst): f'Freq: {data["Frequency"]}\nAvg Time: {data["Average_time"]}' for src, dst, data in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, font_color='black')

        plt.axis('off')
        plt.show()

    def textualize_graph(self, save_name, input_edges, input_nodes):
        nodes = {}  # dict
        edges = []  # list
        for index, row in input_edges.iterrows():
            src, dst, freq, avg = row.iloc[0].split(';')
            src = src.strip()
            dst = dst.strip()
            avg = round(float(avg))
            if src not in nodes:
                nodes[src] = len(nodes)  # id of node
            if dst not in nodes:
                nodes[dst] = len(nodes)  # id of node
            edges.append({'Source_id': nodes[src], 'Frequency': freq, 'Average_time': avg, 'Destination_id': nodes[dst]})
        nodes = pd.DataFrame(nodes.items(), columns=['node_name', 'node_id'])
        edges = pd.DataFrame(edges)

        nodes_df = pd.DataFrame.from_dict(input_nodes, orient='index').reset_index()

        # insert a column to the dataframe that contains the different activities
        nodes_df.columns = ['node_name'] + list(nodes_df.columns)[1:]

        # Merge the nodes DataFrame with the JSON DataFrame
        detailed_nodes = pd.merge(nodes, nodes_df, on='node_name', how='left')

        # create csv files that contained the textualized nodes and edges for embedding
        detailed_nodes.to_csv(f'{save_name}/nodes_textualized.csv', index=False, columns=detailed_nodes.columns)
        edges.to_csv(f'{save_name}/edges_textualized.csv', index=False, columns=['Source_id', 'Frequency', 'Average_time', 'Destination_id'])
        self.set_edges(edges)
        self.set_nodes(detailed_nodes)

    def embed_graph(self, save_name, embedder):
        nodes = pd.read_csv(f"{save_name}/nodes_textualized.csv")
        edges = pd.read_csv(f"{save_name}/edges_textualized.csv")

        # depending on the type of the node attribute, embed it accordingly
        embedded_node_attrs = []
        for col in nodes.columns[1:]:
            if pd.api.types.is_numeric_dtype(nodes[col]):
                node_attr_emb = embedder.embed(nodes[col].astype(str).tolist())
            else:
                node_attr_emb = embedder.embed(nodes[col].tolist())
            embedded_node_attrs.append(node_attr_emb)

        # concatenate / combine the node embeddings
        x = torch.cat(embedded_node_attrs, dim=1)

        # embedding the different edge attributes seperately
        edges_freq_emb = embedder.embed(edges["Frequency"].astype(str).tolist())
        edges_avg_time_emb = embedder.embed(edges["Average_time"].astype(str).tolist())

        # concatenate / combine the edge embeddings
        e = torch.cat([edges_freq_emb, edges_avg_time_emb], dim=1)

        # create the edge index tensor
        edge_index = torch.LongTensor([edges.Source_id, edges.Destination_id])

        # create a data object that represents the knowledge graph
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

        # save the data object to a file
        torch.save(data, f'{save_name}/graphs.pt')
        self.graph = data

    def retrieve_subgraph_pcst(self, question, embedder, topk=3, topk_e=3, cost_e=0.5):

        q_emb_nodes = embedder.embed(question, int(self.graph.x.shape[1]/1024))
        q_emb_edges = embedder.embed(question, int(self.graph.edge_attr.shape[1]/1024))

        c = 0.01
        if len(self.nodes) == 0 or len(self.edges) == 0:
            desc = self.nodes.to_csv(index=False) + '\n' + self.edges.to_csv(index=False,
                                                                                   columns=['Source_id', 'Frequency',
                                                                                            'Average_time', 'Destination_id'])
            graph = Data(x=self.graph.x, edge_index=self.graph.edge_index, edge_attr=self.graph.edge_attr, num_nodes=self.graph.num_nodes)
            return graph, desc

        root = -1  # unrooted
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0
        if topk > 0:
            n_prizes = torch.nn.functional.cosine_similarity(q_emb_nodes, self.graph.x)
            topk = min(topk, self.graph.num_nodes)
            _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

            n_prizes = torch.zeros_like(n_prizes)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
        else:
            n_prizes = torch.zeros(self.graph.num_nodes)

        if topk_e > 0:
            e_prizes = torch.nn.functional.cosine_similarity(q_emb_edges, self.graph.edge_attr)
            topk_e = min(topk_e, e_prizes.unique().size(0))

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e - k) / sum(indices), last_topk_e_value)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
        else:
            e_prizes = torch.zeros(self.graph.num_edges)

        costs = []
        edges = []
        virtual_n_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        for i, (src, dst) in enumerate(self.graph.edge_index.T.numpy()):
            prize_e = e_prizes[i]
            if prize_e <= cost_e:
                mapping_e[len(edges)] = i
                edges.append((src, dst))
                costs.append(cost_e - prize_e)
            else:
                virtual_node_id = self.graph.num_nodes + len(virtual_n_prizes)
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

        selected_nodes = vertices[vertices < self.graph.num_nodes]
        selected_edges = [mapping_e[e] for e in edges if e < num_edges]
        virtual_vertices = vertices[vertices >= self.graph.num_nodes]
        if len(virtual_vertices) > 0:
            virtual_vertices = vertices[vertices >= self.graph.num_nodes]
            virtual_edges = [mapping_n[i] for i in virtual_vertices]
            selected_edges = np.array(selected_edges + virtual_edges)

        edge_index = self.graph.edge_index[:, selected_edges]
        selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

        n = self.nodes.iloc[selected_nodes]
        e = self.edges.iloc[selected_edges]
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['Source_id', 'Frequency', 'Average_time', 'Destination_id'])

        mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

        x = self.graph.x[selected_nodes]
        edge_attr = self.graph.edge_attr[selected_edges]
        src = [mapping[i] for i in edge_index[0].tolist()]
        dst = [mapping[i] for i in edge_index[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))
        subgraph = Graph(description=desc, graph=data, edges=e, nodes=n)
        return subgraph

    def process_graph_data(self):
        # Detailed processing of graph data can be added here.
        graph_list = list()
        graph_list.append(f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges.\n")
        graph_list.append("Here are the nodes in detail:\n")
        column_names = self.nodes.columns
        for index, node in self.nodes.iterrows():
            node_string = f"The index of the node is {node.iloc[1]} and the nodes name is {node.iloc[0]}. It has those attributes: "
            for i in range(2,len(column_names)):
                node_string += f"{column_names[i]}: {node.iloc[i]},"
            node_string = node_string[:-1]
            node_string += ".\n"
            graph_list.append(node_string)
        graph_list.append("Here are the edges in detail:\n")
        edge_names = self.edges.columns
        for index, edge in self.edges.iterrows():
            edge_string = f"There is an edge from the node with index {edge.iloc[0]} to the node with index {edge.iloc[3]} and it has the attributes: "
            for i in range(1,len(edge_names)-1):
                edge_string += f"{edge_names[i]}: {edge.iloc[i]},"
            edge_string = edge_string[:-1]
            edge_string += ".\n"
            graph_list.append(edge_string)
        return "".join(graph_list)