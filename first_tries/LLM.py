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
import numpy as np
import time


class LLM:
    def __init__(self, model_name="meta-llama/Llama-2-13b-chat-hf", **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")

        generation_config = GenerationConfig.from_pretrained(model_name)
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

    def __call__(self, q, graph=None, previous_conversation=None):
        graph_info = graph.process_graph_data() if graph else ""
        prompt_text = self.create_prompt(q, graph_info, previous_conversation)
        return self.hf_pipeline(prompt_text)

    def create_prompt(self, question, graph_info, previous_conversation):
        template = """
        <s>[INST] <<SYS>>
        Act as a Data Scientist who works in Process Analysis and talks to another Data Scientist who.
        For the answer use the graph and all the information that come with it. If your answer contains some of the nodes, always use the nodes name instead of the index.
        Sometimes the answer can refer to the previous answer by you and build on that, consider that.       
        The answer should always contain full sentences with bulletpoints when appropiate. The general conversation language should be business style and your conversation partner should be treated that way. 
  
        <</SYS>>
        Here is the content of the previous conversation. It always consists of an input graph, a question and your answer:
        {prev_conv}
        
        Now here is the question for you to answer:
        {q}
        
        And to answer the question consider this graph:
        {graph}      
        [/INST]
        """
        prompt = PromptTemplate(input_variables=["prev_conv","q","graph"], template=template)
        prev_conv = ""
        graph = ""
        if previous_conversation:
            prev_conv = previous_conversation
        if graph_info:
            graph = graph_info
        return prompt.format(prev_conv=prev_conv,q=question,graph=graph)

class TextEmbedder:
    def __init__(self, model='sbert'):
        self.model, self.tokenizer, self.device = load_model[model]()
        self.text2embedding = load_text2embedding[model]

    def embed(self, input, multiplier=1):
        embedding = self.text2embedding(self.model, self.tokenizer, self.device, input)
        return embedding.repeat(1, multiplier)

    def get_query_embedding(query, repeat_factor, tokenizer, model):
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            query_emb = outputs.last_hidden_state.mean(dim=1)
        query_emb = query_emb.repeat(1, repeat_factor)  # Repeat the embedding to match the graph dimension
        return query_emb

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

        for i, row in self.edges.iterrows():
            g.add_edge(row['src'], row['dst'], frequency=row['freq'], avg_time=row['avg_time'])

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
            src, dst, freq, avg = row.iloc[0].split(';')
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

    def retrieve_subgraph_pcst(self, question, embedder, topk=3, topk_e=3, cost_e=0.5):

        q_emb_nodes = embedder.embed(question, 3)
        q_emb_edges = embedder.embed(question, 2)

        c = 0.01
        if len(self.nodes) == 0 or len(self.edges) == 0:
            desc = self.nodes.to_csv(index=False) + '\n' + self.edges.to_csv(index=False,
                                                                                   columns=['src', 'frequency',
                                                                                            'avg_time', 'dst'])
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
        desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'freq', 'avg_time', 'dst'])

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

class Conversation:
    def __init__(self):
        self.llm = LLM()
        self.prev_conv = dict()

    def ask_question(self, graph, question):
        text_g = graph.process_graph_data()
        prompt = self.llm.create_prompt(question,text_g,self.textualize_prev_conf())
        print("Asking question, wait for response...")
        start_time = time.time()
        answer = self.llm(prompt)
        print(answer)
        end_time = time.time()
        print(f"The generation of this answer took {(end_time-start_time):.4f} seconds")
        graph.visualize_graph()
        self.prev_conv[len(self.prev_conv)+1] = {'question': question, 'text_g':text_g, 'answer':answer}

    def textualize_prev_conf(self):
        text_prev_conf =[]
        for key, value in self.prev_conv.items():
            conv_string = f"The {key}. question was '{value['question']}' and the inputted graph looks like this '{value['text_g']}' and the answer you created is the following '{value['answer']}'\n"
            text_prev_conf.append(conv_string)
        return "".join(text_prev_conf)

if __name__ == '__main__':
    save_folder='test'
    os.makedirs(f'{save_folder}', exist_ok=True)
    log = pm4py.read_xes("payment_process_vet.xes")
    event_log = EventLog(log)
    event_log.preprocess_log()
    know_g = event_log.create_kg(save_folder)
    embedder = TextEmbedder()
    know_g.embed_graph(save_folder, embedder)
    input_question = "What are the edges with the highest Frequency?"
    subgraph = know_g.retrieve_subgraph_pcst(input_question, embedder)
    conv = Conversation()
    conv.ask_question(subgraph, input_question)
    # Can ask a second question like this:
    # second_question = "What would be the most important edges in the Graph?"
    # subgraph = know_g.retrieve_subgraph_pcst(second_question, embedder)
    # conv.ask_question(subgraph, second_question)



