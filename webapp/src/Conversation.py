import time
from . import LLM
import json
import os
import psutil

class Conversation:
    def __init__(self, know_g, llm = "meta-llama/Llama-2-7b-chat-hf", mode="local", gpt_model="gpt-3.5-turbo-16k"):
        self.llm_params = {
            'llm': llm,
            'mode': mode,
            'gpt_model': gpt_model
        }
        self.llm = LLM.LLM(mode=mode, gpt_model=gpt_model, model_name=llm)
        self.prev_conv = dict()
        self.know_g = know_g

    def ask_question(self, graph, question):
        text_g = graph.process_graph_data()
        print("Asking question, wait for response...")
        start_time = time.time()
        answer = self.llm(q=question, graph_info=text_g, previous_conversation=self.textualize_prev_conf())
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
        conv_data[len(self.prev_conv)] = {
            'question': question,
            'answer': self.prev_conv[len(self.prev_conv)]['answer']
        }

        # Save the updated content back to the file
        with open(file_path, "w") as json_file:
            json.dump(conv_data, json_file, indent=4)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the LLM instance
        state['llm'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the LLM instance using saved parameters
        llm_params = state.get('llm_params', {})
        self.llm = LLM.LLM(
            mode=llm_params.get('mode', 'local'),
            gpt_model=llm_params.get('gpt_model', 'gpt-3.5-turbo-16k'),
            model_name=llm_params.get('llm', 'meta-llama/Llama-2-7b-chat-hf')
        )

def check_available_ram():
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert bytes to GB
    print(f"Available RAM: {available_ram:.2f} GB")
    return available_ram

if __name__ == '__main__':
    print('That´s not how you call this file')