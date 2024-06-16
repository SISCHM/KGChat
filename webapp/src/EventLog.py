import json
import pandas as pd
import pm4py
from . import Graph
import os
import warnings
from lxml import etree

class EventLog:
    def __init__(self,file_path):
        warnings.filterwarnings("ignore")
        if file_path.endswith(".xes"):
            self.log = pm4py.read_xes(file_path, disable_progress_bar= True)
        else:
            raise ValueError("Unsupported file format")

        self.dfg = None
        self.start_activities = None
        self.end_activities = None
        self.performance_dfg = None
        self.name = os.path.splitext(os.path.basename(file_path))[0]

    def preprocess_log(self, selected_columns):
        # Ask the user to select attributes of the events which they want to keep for the knowledge graph
        self.log = self.log[selected_columns]
        for col in selected_columns:
            # if column is of numeric type, then replace NaN with 0
            if pd.api.types.is_numeric_dtype(self.log[col]):
                self.log[col] = self.log[col].fillna(0)
            else:
                self.log[col] = self.log[col].fillna("Unknown")

    def create_edges(self):
        with open(f"chats/{self.name}/edges.tsv", "w", newline='') as file:
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

    def create_nodes(self):
        # grouped the log by the activity name and for each activity, compute the average value of every numerical attribute or create a list of unique values for every non-numerical attribute 
        columns = self.log.columns.tolist()
        aggregation = {col: ('mean' if pd.api.types.is_numeric_dtype(self.log[col]) else lambda x: list(x.unique())) for col in columns if col not in ["concept:name", "case:concept:name"] and "timestamp" not in col}
        log = self.log.groupby("concept:name").agg(aggregation).reset_index()

        # create a dict with every key being an unique activity and its values being the relevant attributes and values
        columns = log.columns.tolist()
        initial_dict = {}
        for _, row in log.iterrows():
            event_attributes = {}
            for col in columns:
                if pd.api.types.is_numeric_dtype(log[col]):
                    event_attributes[f"Average_{col}"] = row[col]
                else:
                    event_attributes[col] = row[col]
            initial_dict[row["concept:name"]] = event_attributes

        # create and store two nodes for start and end activities with the default attributes and values in the dict
        start_end_defaults = {(f"Average_{col}" if pd.api.types.is_numeric_dtype(self.log[col]) else col): (0.0 if pd.api.types.is_numeric_dtype(log[col]) else ["Unknown"]) for col in columns}

        # add artificial start and end nodes
        res_dict = {
            "start": start_end_defaults
        }

        res_dict.update(initial_dict)

        res_dict["end"] = start_end_defaults

        # convert the dict to a json object and then write it to a json file
        with open(f"chats/{self.name}/nodes.json", "w") as json_file:
            json.dump(res_dict, json_file, indent=4)

    def construct_dfg(self):
        self.dfg, self.start_activities, self.end_activities = pm4py.discover_directly_follows_graph(self.log)
        self.performance_dfg, _, _ = pm4py.discover_performance_dfg(self.log)

    def create_kg(self):
        return self.create_knowledge_graph()

    def create_knowledge_graph(self):
        self.construct_dfg()
        os.makedirs(f'chats/{self.name}', exist_ok=True)
        os.makedirs(f'chats/{self.name}/graphs', exist_ok=True)
        self.create_edges()
        self.create_nodes()
        knowledge_g = Graph.Graph()
        with open(f"chats/{self.name}/nodes.json", "r") as file:
            nodes = json.load(file)
        edges = pd.read_csv(f"chats/{self.name}/edges.tsv", sep='\t')
        knowledge_g.textualize_graph(self.name, edges, nodes)
        return knowledge_g

def extract_columns_from_xes(file_path):
    # Parse the XML file
    tree = etree.parse(file_path)
    root = tree.getroot()

    # Find all unique attributes (columns) in the first trace
    columns = set()
    for trace in root.findall('trace'):
        for event in trace.findall('event'):
            for attribute in event.findall('*'):
                columns.add(attribute.get('key'))
        break  # Only need the first trace for column extraction
    columns.add("case:concept:name")
    return list(columns)

if __name__ == '__main__':
    print('That´s not how you call this file')
