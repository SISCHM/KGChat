import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

class TextEmbedding:
    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="thenlper/gte-large",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        ) # based on Bert
    def embed_query(self, input):
        return self.embedding.embed_query(input)

class TextSplitter:
    def __init__(self):
        self.chunk_size = 1024
        self.chunk_overlap = 64
        self.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def split_documents(self, input_docs):
        return self.RecursiveCharacterTextSplitter(input_docs)



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