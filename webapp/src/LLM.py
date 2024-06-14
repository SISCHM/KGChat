import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings
import os
import openai

def get_huggingface_token():
    current_dir = os.path.dirname(__file__)
    token_file_path = os.path.join(current_dir, 'utils', 'HUGGINGFACE_TOKEN.txt')
    try:
        with open(token_file_path, "r") as file:
            token = file.read().strip()
    except FileNotFoundError:
        token = input("Please enter your Hugging Face API token: ").strip()
        with open(token_file_path, "w") as file:
            file.write(token)
    return token


def get_openai_api_key():
    current_dir = os.path.dirname(__file__)
    key_file_path = os.path.join(current_dir, 'utils', 'OPENAI_API_KEY.txt')
    try:
        with open(key_file_path, "r") as file:
            key = file.read().strip()
    except FileNotFoundError:
        key = input("Please enter your OpenAI API token: ").strip()
        with open(key_file_path, "w") as file:
            file.write(key)
    return key


def create_prompt(question, graph_info, previous_conversation):
    template = """
    <s>[INST] <<SYS>>
    Act as a Data Scientist who works in Process Analysis and talks to another Data Scientist.
    For the answer use the graph and all the information that come with it. If your answer contains some of the nodes, always use the nodes name instead of the index and do not mention the indexes, also when talking about the edges use the corresponding names of the nodes instead of their indexes.
    Sometimes the answer can refer to the previous answer by you and build on that, consider that.       
    The answer should always contain full sentences with bulletpoints when appropriate. The general conversation language should be business style and your conversation partner should be treated that way. 

    <</SYS>>
    Here is the content of the previous conversation. It always consists of an input graph, a question and your answer:
    {prev_conv}

    Now here is the question for you to answer:
    {q}

    And to answer the question consider this graph:
    {graph}      
    [/INST]
    """
    prompt = PromptTemplate(input_variables=["prev_conv", "q", "graph"], template=template)
    prev_conv = previous_conversation if previous_conversation else ""
    graph = graph_info if graph_info else ""
    return prompt.format(prev_conv=prev_conv, q=question, graph=graph)


def create_messages(question, graph, previous_conversation):
    messages = [
        {"role": "system", "content": "Act as a Data Scientist who works in Process Analysis and talks to another Data Scientist. For the answer use the graph and all the information that come with it. If your answer contains some of the nodes, always use the nodes name instead of the index and do not mention the indexes, also when talking about the edges use the corresponding names of the nodes instead of their indexes. Sometimes the answer can refer to the previous answer by you and build on that, consider that. The answer should always contain full sentences with bulletpoints when appropriate. The general conversation language should be business style and your conversation partner should be treated that way. The answer should be short but precise. The answer should not contain 'Based on the information provided in the graph'. Provide direct and concise answers."}
    ]
    if previous_conversation:
        messages.append({"role": "user", "content": previous_conversation})
    if graph:
        messages.append({"role": "user", "content": f"Graph information: {graph}"})
    messages.append({"role": "user", "content": f"Here ist the question: {question}"})
    return messages


class LLM:
    def __init__(self, mode='local', model_name="meta-llama/Llama-2-7b-chat-hf", gpt_model="gpt-3.5-turbo-16k", **kwargs):
        self.mode = mode.lower()
        if self.mode == 'local':
            self._setup_local_model(model_name, **kwargs)
        elif self.mode == 'remote':
            self._setup_remote_model(gpt_model, **kwargs)
        else:
            raise ValueError("Mode must be 'local' or 'remote'.")

    def _setup_local_model(self, model_name, **kwargs):
        token = get_huggingface_token()
        warnings.filterwarnings("ignore")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True,
                                                     device_map="auto", use_auth_token=token)

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
        self.model = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.01})

    def _setup_remote_model(self, gpt_model, **kwargs):
        self.gpt_model = gpt_model
        self.model = openai.OpenAI(api_key=get_openai_api_key())

    def __call__(self, q, graph_info=None, previous_conversation=None):
        if self.mode == 'local':
            prompt_text = create_prompt(q, graph_info, previous_conversation)
            return self.model(prompt_text)
        elif self.mode == 'remote':
            messages = create_messages(q, graph_info, previous_conversation)
            response = self.model.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content


class TextSplitter:
    def __init__(self):
        self.chunk_size = 1024
        self.chunk_overlap = 64
        self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                                             chunk_overlap=self.chunk_overlap)

    def split_documents(self, input_docs):
        return self.RecursiveCharacterTextSplitter(input_docs)


if __name__ == '__main__':
    print('That´s not how you call this file')
