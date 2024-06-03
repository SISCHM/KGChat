import pm4py 
import json
import sys
import pandas as pd


def read_event_log(file_path):

    if file_path.endswith(".xes"):
        log = pm4py.read_xes(file_path)
        
    else:
        raise ValueError("Unsupported file format")

    return log


# Replace NaN values with appropriate default values 
def preprocess_log(log, selected_cols):
    for col in selected_cols:
        # if column is of numeric type, then replace NaN with 0 
        if pd.api.types.is_numeric_dtype(log[col]):
            log[col] = log[col].fillna(0)
        else:
            log[col] = log[col].fillna("Unknown")
    return log


# create a tsv file which saves the details of the edges of the knowledge graph constructed from an event log 
def write_edges_to_tsv(log) :

    #construct DFGs with frequency and performance metrics on the edges
    dfg, start_acts, end_acts = pm4py.discover_directly_follows_graph(log)
    perf_dfg, _, _ = pm4py.discover_performance_dfg(log)
    
    # write information on the edges, represented by tuples, into a tsv file 
    with open("edges.tsv", "w", newline='') as file:

        file.write("graph\n")

        # start activity is connected to a default "start" node as its source node
        for start_act, freq in start_acts.items():
            file.write(f"(start; {start_act}; {freq}; 0)")

        # for every pair of activities which are connected by a directly-follows relationship
        for (source, target), freq in dfg.items():
            for (pf_source, pf_target), metrics in perf_dfg.items(): 
                 if pf_source == source and pf_target == target:
                     # every tuple represents an edge between a source node to a target node with the frequency and average time also recorded
                     file.write(f"({source}; {target}; {freq}; {metrics.get('mean','')})")
        
        # end activity is connected to a default "end" node as its target node
        for end_act, freq in end_acts.items():
            file.write(f"({end_act}; end; {freq}; 0)")


# create a json file that stores every activity and its selected attributes 
def write_nodes_to_json(log, selected_columns):
    
    # define aggregation for numerical attributes and non-numerical attributes (by default, num attr: mean, non-num: list of unique values)
    aggregation = {col: ('mean' if pd.api.types.is_numeric_dtype(log[col]) else lambda x: list(x.unique())) for col in selected_columns}

    # group the log by the activity name and apply the aggregation 
    log_grouped = log.groupby("concept:name").agg(aggregation).reset_index()

    # create a dict with every key being an unique activity and its values being its relevant attributes and values 
    initial_dict = {}
    for _, row in log_grouped.iterrows():
        event_attributes = {}
        for col in selected_columns:
            if pd.api.types.is_numeric_dtype(log[col]):
                event_attributes[f"avg_{col}"] = row[col]
            else:
                event_attributes[col] = row[col]
        initial_dict[row["concept:name"]] = event_attributes

    # create and store two nodes for start and end activities with the default attributes and values in the dict  
    start_end_defaults = {
        (f"avg_{col}" if pd.api.types.is_numeric_dtype(log[col]) else col): (0.0 if pd.api.types.is_numeric_dtype(log[col]) else ["Unknown"])
        for col in selected_columns
    }

    res_dict = {
        "start": start_end_defaults
    }

    res_dict.update(initial_dict)

    res_dict["end"] = start_end_defaults

    # convert the dict to a json object and then write it to a json file 
    with open("nodes.json", "w") as json_file:
        json.dump(res_dict, json_file, indent=4)

        
if __name__ == '__main__':
    
    if len(sys.argv) < 2 :
        print("Please include an xes event log file")
        sys.exit(1)

    event_log_file = sys.argv[1]
    log = read_event_log(event_log_file)

    # Ask the user to select attributes of the events which they want to keep for the knowledge graph
    all_columns = log.columns.tolist()
    selected_columns = []
    print("Select attributes from the event log to be included in the knowledge graph:")
    for col in all_columns:
        keep = input(f"Keep attribute '{col}'? (y/n): ").strip().lower()
        if keep == 'y':
            selected_columns.append(col)
    
    filtered_log = preprocess_log(log, selected_columns)

    write_edges_to_tsv(filtered_log)
    write_nodes_to_json(filtered_log, selected_columns)