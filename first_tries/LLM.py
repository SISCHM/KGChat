import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings



class LLM:
    def __init__(self, model_name="meta-llama/Llama-2-13b-chat-hf", **kwargs):
        warnings.filterwarnings("ignore")
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

class TextSplitter:
    def __init__(self):
        self.chunk_size = 1024
        self.chunk_overlap = 64
        self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def split_documents(self, input_docs):
        return self.RecursiveCharacterTextSplitter(input_docs)

if __name__ == '__main__':
    print('That´s not how you call this file')



