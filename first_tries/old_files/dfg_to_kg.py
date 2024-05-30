import pm4py 
import json

log = pm4py.read_xes("../payment_process_vet.xes")

# create a tsv file which saves the details of the edges of the knowledge graph constructed from an event log 
def function1() :

    #construct DFGs with frequency and performance metrics on the edges
    dfg, start_acts, end_acts = pm4py.discover_directly_follows_graph(log)
    perf_dfg, perf_start_acts, perf_end_acts = pm4py.discover_performance_dfg(log)
    
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

def function2():
    
    # Preprocessing the event log by replacing NaN values with relevant values
    log["cost"] = log["cost"].fillna(0)
    log["org:resource"] = log["org:resource"].fillna("External")
    log["treatment"] = log["treatment"].fillna("No treatment involved")
    log["invoicedPrice"] = log["invoicedPrice"].fillna(0)

    # Grouped the log by the activity name and for each activity compute the avg cost, avg invoiced price and resources involved for all its instances
    log_grouped = log.groupby("concept:name").agg({"cost": "mean", 
                                                   "invoicedPrice": "mean",
                                                   "org:resource": lambda x: list(x.unique())})

    # Convert the the "concept:name" column back to a normal column 
    log_grouped = log_grouped.reset_index()

    # Create a dict with every key being an unique activity and its values being its relevant attributes and values 
    initial_dict = {
        row["concept:name"]: {
            "average_cost": row["cost"],
            "average_invoiced_price": row["invoicedPrice"],
            "resources_involved": row["org:resource"]
        }
        for _, row in log_grouped.iterrows()
    }

    res_dict = {"start": {
        "average_cost": 0.0,
        "average_invoiced_price": 0.0,
        "resources_involved": [
            "External"
        ]
        }
    }

    res_dict.update(initial_dict)

    res_dict["end"] = {
        "average_cost": 0.0,
        "average_invoiced_price": 0.0,
        "resources_involved": [
            "External"
        ]
    }

    # convert the dict to a json object and then write it to a json file 
    with open("nodes.json", "w") as json_file:
        json.dump(res_dict, json_file, indent=4)

        
if __name__ == '__main__':
    function1()
    function2()