import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import networkx as nx
import matplotlib.pyplot as plt
from pcst_fast import pcst_fast
import pm4py
import pandas as pd
from src.utils.lm_modeling import load_model, load_text2embedding
import os
import json
from torch_geometric.data import Data

MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"

class LLM:
    def __init__(self, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")

        generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.01
        generation_config.top_p = 0.95
        generation_config.do_sample = True
        generation_config.repetition_penalty = 1.15

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
        self.hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.01})

    def __call__(self, prompt_text):
        return self.hf_pipeline(prompt_text)

class TextEmbedder:
    def __init__(self, model='sbert'):
        self.model, self.tokenizer, self.device = load_model[model]()
        self.text2embedding = load_text2embedding[model]

    def embed(self, input):
        return self.text2embedding(self.model, self.tokenizer, self.device, input)

class TextSplitter:
    def __init__(self):
        self.chunk_size = 1024
        self.chunk_overlap = 64
        self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def split_documents(self, input_docs):
        return self.RecursiveCharacterTextSplitter(input_docs)

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

    def visualize_graph(self):
        g = nx.DiGraph()

        for i, node in self.nodes.iterrows():
            g.add_node(i, label=node['node_attr'], average_cost=node['average_cost'], average_invoiced_price=node['average_invoiced_price'])

        for i, (src, dst) in enumerate(self.graph.edge_index.t().tolist()):
            g.add_edge(src, dst, frequency=self.graph.edge_attr[i][0].item(),
                           avg_time=self.graph.edge_attr[i][1].item())

        pos = nx.spring_layout(g)
        labels = nx.get_node_attributes(g, 'label')
        edge_labels = {(src, dst): f'Freq: {data["frequency"]}\nAvg Time: {data["avg_time"]}' for src, dst, data in
                       g.edges(data=True)}

        nx.draw(g, pos, labels=labels, with_labels=True, node_size=500, node_color="lightblue", font_size=8,
                font_weight="bold", arrows=True)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)
        plt.show()

    def textualize_graph(self, save_name, input_edges, input_nodes):
        nodes = {}  # dict
        edges = []  # list
        for index, row in input_edges.iterrows():
            src, dst, freq, avg = row[0].split(';')
            src = src.strip()
            dst = dst.strip()
            avg = round(float(avg))
            if src not in nodes:
                nodes[src] = len(nodes)  # id of node
            if dst not in nodes:
                nodes[dst] = len(nodes)  # id of node
            edges.append({'src': nodes[src], 'freq': freq, 'avg_time': avg, 'dst': nodes[dst]})
        nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
        edges = pd.DataFrame(edges)


        nodes_df = pd.DataFrame.from_dict(input_nodes, orient='index').reset_index()

        # insert a column to the dataframe that contains the different activites
        nodes_df.columns = ['node_attr'] + list(nodes_df.columns)[1:]

        # Merge the nodes DataFrame with the JSON DataFrame
        detailed_nodes = pd.merge(nodes, nodes_df, on='node_attr', how='left')

        # create csv files that contained the textualized nodes and edges for embedding
        detailed_nodes.to_csv(f'{save_name}/nodes_textualized.csv', index=False, columns=detailed_nodes.columns)
        edges.to_csv(f'{save_name}/edges_textualized.csv', index=False, columns=['src', 'freq', 'avg_time', 'dst'])
        self.set_edges(edges)
        self.set_nodes(detailed_nodes)

    def embed_graph(self, save_name, embedder):
        print('Encoding graphs...')
        nodes = pd.read_csv(f"{save_name}/nodes_textualized.csv")
        edges = pd.read_csv(f"{save_name}/edges_textualized.csv")

        # embedding the different node attributes seperately
        nodes_attr_emb = embedder.embed(nodes["node_attr"].tolist())
        nodes_avg_cost_emb = embedder.embed(nodes["average_cost"].astype(str).tolist())
        nodes_avg_inv_emb = embedder.embed(nodes["average_invoiced_price"].astype(str).tolist())

        # concatenate / combine the node embeddings
        x = torch.cat([nodes_attr_emb, nodes_avg_cost_emb, nodes_avg_inv_emb], dim=1)

        # embedding the different edge attributes seperately
        edges_freq_emb = embedder.embed(edges["freq"].astype(str).tolist())
        edges_avg_time_emb = embedder.embed(edges["avg_time"].astype(str).tolist())

        # concatenate / combine the edge embeddings
        e = torch.cat([edges_freq_emb, edges_avg_time_emb], dim=1)

        # create the edge index tensor
        edge_index = torch.LongTensor([edges.src, edges.dst])

        # create a data object that represents the knowledge graph
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))

        # save the data object to a file
        torch.save(data, f'{save_name}/graphs.pt')
        self.graph = data

    def retrieve_subgraph_pcst(self, emb_question, topk=3, topk_e=3, cost_e=0.5):
        c = 0.01
        if len(self.nodes) == 0 or len(self.edges) == 0:
            desc = self.nodesto_csv(index=False) + '\n' + self.edges.to_csv(index=False,columns=['src', 'frequency', 'avg_time', 'dst'])
            graph = Data(x=self.graph.x, edge_index=self.graph.edge_index, edge_attr=self.graph.edge_attr, num_nodes=self.graph.num_nodes)
            return graph, desc

        root = -1  # unrooted
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0

        print("emb_question shape:", emb_question.shape)
        print("self.graph.x shape:", self.graph.x.shape)
        print("self.graph.edge_attr shape:", self.graph.edge_attr.shape)

        if topk > 0:
            n_prizes = torch.nn.CosineSimilarity(dim=-1)(emb_question, self.graph.x)
            topk = min(topk, self.graph.num_nodes)
            _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

            n_prizes = torch.zeros_like(n_prizes)
            n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
        else:
            n_prizes = torch.zeros(self.graph.num_nodes)

        if topk_e > 0:
            e_prizes = torch.nn.CosineSimilarity(dim=-1)(emb_question, self.graph.edge_attr)
            topk_e = min(topk_e, e_prizes.unique().size(0))

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e - k) / sum(indices), last_topk_e_value)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
            # reduce the cost of the edges such that at least one edge is selected
            # cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
        else:
            e_prizes = torch.zeros(self.graph.num_edges)

        costs = []
        edges = []
        vritual_n_prizes = []
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
                virtual_node_id = self.graph.num_nodes + len(vritual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                vritual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
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
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'frequency', 'avg_time', 'dst'])

        mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

        x = self.graph.x[selected_nodes]
        edge_attr = self.graph.edge_attr[selected_edges]
        src = [mapping[i] for i in edge_index[0].tolist()]
        dst = [mapping[i] for i in edge_index[1].tolist()]
        edge_index = torch.LongTensor([src, dst])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))
        subgraph = Graph(description=desc, graph=data, edges=e, nodes=n)
        return subgraph

class EventLog:
    def __init__(self,log):
        self.log = log
        self.dfg = None
        self.start_activities = None
        self.end_activities = None
        self.performance_dfg = None

    def preprocess_log(self):
        self.log["cost"] = self.log["cost"].fillna(0)
        self.log["org:resource"] = self.log["org:resource"].fillna("External")
        self.log["treatment"] = self.log["treatment"].fillna("No treatment involved")
        self.log["invoicedPrice"] = self.log["invoicedPrice"].fillna(0)

    def create_edges(self, save_name):
        with open(f"{save_name}/edges.tsv", "w", newline='') as file:
            # every tuple represents an edge between a source node to a target node with the frequency and average time also recorded
            file.write("Source;Target;Frequency;Average Time\n")

            # start activity is connected to a default "start" node as its source node
            for start_act, freq in self.start_activities.items():
                file.write(f"start;{start_act};{freq};0\n")

            # for every pair of activities which are connected by a directly-follows relationship
            for (source, target), freq in self.dfg.items():
                for (pf_source, pf_target), metrics in self.performance_dfg.items():
                    if pf_source == source and pf_target == target:
                        file.write(f"{source};{target};{freq};{metrics.get('mean', '')}\n")

            # end activity is connected to a default "end" node as its target node
            for end_act, freq in self.end_activities.items():
                file.write(f"{end_act};end;{freq};0\n")

    def create_nodes(self, save_name):
        # Grouped the log by the activity name and for each activity compute the avg cost, avg invoiced price and resources involved for all its instances
        log_grouped = self.log.groupby("concept:name").agg({"cost": "mean",
                                                       "invoicedPrice": "mean",
                                                       "org:resource": lambda x: list(x.unique())})

        # Convert the the "concept:name" column back to a normal column
        log_grouped = log_grouped.reset_index()

        # Create a dict with every key being an unique activity and its values being its relevant attributes and values
        res_dict = {
            row["concept:name"]: {
                "average_cost": row["cost"],
                "average_invoiced_price": row["invoicedPrice"],
                "resources_involved": row["org:resource"]
            }
            for _, row in log_grouped.iterrows()
        }
        # add artificial start and end nodes
        res_dict['start'] = {"average_cost": 0.0, "average_invoiced_price": 0.0, "resources_involved": []}
        res_dict['end'] = {"average_cost": 0.0, "average_invoiced_price": 0.0, "resources_involved": []}

        # convert the dict to a json object and then write it to a json file
        with open(f"{save_name}/nodes.json", "w") as json_file:
            json.dump(res_dict, json_file, indent=4)

    def construct_dfg(self):
        self.dfg, self.start_activities, self.end_activities = pm4py.discover_directly_follows_graph(self.log)
        self.performance_dfg, _, _ = pm4py.discover_performance_dfg(self.log)

    def create_kg(self, save_name):
        return self.create_knowledge_graph(save_name)

    def create_knowledge_graph(self, save_name):
        self.construct_dfg()
        self.create_edges(save_name)
        self.create_nodes(save_name)
        knowledge_g = Graph()
        with open(f"{save_name}/nodes.json", "r") as file:
            nodes = json.load(file)
        edges = pd.read_csv(f"{save_name}/edges.tsv", sep='\t')
        knowledge_g.textualize_graph(save_name, edges, nodes)
        return knowledge_g



def test():
    template = """
    <s>[INST] <<SYS>>
    Act as a Machine Learning engineer who is teaching high school students.
    <</SYS>>
    
    {text} [/INST]
    """

    text = "Explain what are Deep Neural Networks in 2-3 sentences"
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    test = prompt.format(text=text)
    llm = LLM()
    result = llm(test)
    print(result)

if __name__ == '__main__':
    save_folder='test'
    os.makedirs(f'{save_folder}', exist_ok=True)
    log = pm4py.read_xes("payment_process_vet.xes")
    event_log = EventLog(log)
    event_log.preprocess_log()
    know_g = event_log.create_kg(save_folder)
    embedder = TextEmbedder()
    know_g.embed_graph(save_folder, embedder)
    know_g.visualize_graph()
    question = 'How often gets something payed?'
    emb_question = embedder.embed(question)
    subgraph = know_g.retrieve_subgraph_pcst(emb_question)
    subgraph.visualize_graph()


