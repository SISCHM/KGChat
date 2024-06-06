import time
import LLM
import TextEmbedder
import EventLog
import sys
import json

class Conversation:
    def __init__(self):
        self.llm = LLM.LLM()
        self.prev_conv = dict()

    def ask_question(self, graph, question):
        text_g = graph.process_graph_data()
        prompt = self.llm.create_prompt(question,text_g,self.textualize_prev_conf())
        print("Asking question, wait for response...")
        start_time = time.time()
        answer = "Test"         # self.llm(prompt)
        #print(answer)
        end_time = time.time()
        print(f"The generation of this answer took {(end_time-start_time):.4f} seconds")
        self.prev_conv[len(self.prev_conv)+1] = {'question': question, 'text_g':text_g, 'answer':answer}

    def textualize_prev_conf(self):
        text_prev_conf =[]
        for key, value in self.prev_conv.items():
            conv_string = f"The {key}. question was '{value['question']}' and the inputted graph looks like this '{value['text_g']}' and the answer you created is the following '{value['answer']}'\n"
            text_prev_conf.append(conv_string)
        return "".join(text_prev_conf)

    def question_to_file(self, log, question):
        with open(f"{log.name}/conv.json", "w") as json_file:
            json.dump({len(self.prev_conv)+1: {'question': question, 'answer':self.prev_conv[len(self.prev_conv)]['answer']}}, json_file, indent=4)

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("Please include an xes event log file")
        sys.exit(1)

    event_log_file = sys.argv[1]
    event_log = EventLog.EventLog(event_log_file)
    event_log.preprocess_log()
    know_g = event_log.create_kg()
    know_g.visualize_graph(event_log.name,0)
    embedder = TextEmbedder.TextEmbedder()
    know_g.embed_graph(event_log.name, embedder)
    input_question = "What are the edges with the highest Frequency?"
    subgraph = know_g.retrieve_subgraph_pcst(input_question, embedder)
    subgraph.visualize_graph(event_log.name,1)
    conv = Conversation()
    conv.ask_question(subgraph, input_question)
    conv.question_to_file(event_log, input_question)

    # Can ask a second question like this:
    # second_question = "What would be the most important edges in the Graph?"
    # subgraph = know_g.retrieve_subgraph_pcst(second_question, embedder)
    # conv.ask_question(subgraph, second_question)