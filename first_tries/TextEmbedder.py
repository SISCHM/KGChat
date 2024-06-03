import torch
from src.utils.lm_modeling import load_model, load_text2embedding

class TextEmbedder:
    def __init__(self, model='sbert'):
        self.model, self.tokenizer, self.device = load_model[model]()
        self.text2embedding = load_text2embedding[model]

    def embed(self, input, multiplier=1):
        embedding = self.text2embedding(self.model, self.tokenizer, self.device, input)
        return embedding.repeat(1, multiplier)

    def get_query_embedding(self, query, repeat_factor, tokenizer, model):
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            query_emb = outputs.last_hidden_state.mean(dim=1)
        query_emb = query_emb.repeat(1, repeat_factor)  # Repeat the embedding to match the graph dimension
        return query_emb