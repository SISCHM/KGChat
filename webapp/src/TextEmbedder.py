import torch
from .utils.lm_modeling import load_model, load_text2embedding

class TextEmbedder:
    def __init__(self, model='sbert'):
        self.model, self.tokenizer, self.device = load_model[model]()
        self.text2embedding = load_text2embedding[model]

    def embed(self, input, multiplier=1):
        embedding = self.text2embedding(self.model, self.tokenizer, self.device, input)
        return embedding.repeat(1, multiplier)


if __name__ == '__main__':
    print('That´s not how you call this file')