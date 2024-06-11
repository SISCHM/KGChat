import time
from . import LLM
from . import TextEmbedder
from . import EventLog
import sys
import json
import os
import psutil

class Conversation:
    def __init__(self, know_g, llm = "meta-llama/Llama-2-7b-chat-hf"):
        self.llm = LLM.LLM(llm)
        self.prev_conv = dict()
        self.know_g = know_g

    def ask_question(self, graph, question):
        text_g = graph.process_graph_data()
        prompt = self.llm.create_prompt(question,text_g,self.textualize_prev_conf())
        print("Asking question, wait for response...")
        start_time = time.time()
        answer = "answer" #self.llm(prompt)
        end_time = time.time()
        print(f"The generation of this answer took {(end_time-start_time):.4f} seconds")
        self.prev_conv[len(self.prev_conv)+1] = {'question': question, 'text_g':text_g, 'answer':answer}

    def textualize_prev_conf(self):
        text_prev_conf =[]
        for key, value in self.prev_conv.items():
            conv_string = f"The {key}. question was '{value['question']}' and the inputted graph looks like this '{value['text_g']}' and the answer you created is the following '{value['answer']}'\n"
            text_prev_conf.append(conv_string)
        return "".join(text_prev_conf)

    def question_to_file(self, save_name, question):
        file_path = os.path.join(save_name, "conv.json")
        # Check if the file exists
        if os.path.exists(file_path):
            # Load the existing content
            with open(file_path, "r") as json_file:
                conv_data = json.load(json_file)
        else:
            # Initialize an empty dictionary if the file doesn't exist
            conv_data = {}

        # Add the new question and answer
        conv_data[len(self.prev_conv) + 1] = {
            'question': question,
            'answer': self.prev_conv[len(self.prev_conv)]['answer']
        }

        # Save the updated content back to the file
        with open(file_path, "w") as json_file:
            json.dump(conv_data, json_file, indent=4)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the LLM and TextEmbedder instances
        state['llm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the LLM and TextEmbedder instances
        self.llm = LLM.LLM()

def check_available_ram():
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    return available_ram

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("Please include an xes event log file")
        sys.exit(1)

    required_ram = 24 # GB, adjust as necessary for your model
    available_ram = check_available_ram()
    if available_ram < required_ram:
        print("Warning: Not enough RAM available to load the model.")
    else:
        print("Sufficient RAM available to load the model.")

    event_log_file = sys.argv[1]
    event_log = EventLog.EventLog(event_log_file)
    event_log.preprocess_log(["severityInjury", "case:concept:name", "treatment", "concept:name", "animalClass", "time:timestamp", "payment"])
    know_g = event_log.create_kg()
    know_g.visualize_graph(event_log.name,0)
    embedder = TextEmbedder.TextEmbedder()
    know_g.embed_graph(event_log.name, embedder)
    input_question = "What are the edges with the highest Frequency?"
    subgraph = know_g.retrieve_subgraph_pcst(input_question, embedder)
    subgraph.visualize_graph(event_log.name,1)
    conv = Conversation(know_g)
    conv.ask_question(subgraph, input_question)
    conv.question_to_file("", input_question)

    # Can ask a second question like this:
    second_question = "What would be the most important edges in the Graph?"
    subgraph = know_g.retrieve_subgraph_pcst(second_question, embedder)
    conv.ask_question(subgraph, second_question)