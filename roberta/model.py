import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class QuestionEmbeddingModel(nn.Module):
    def __init__(self, model_name='roberta-base'):
        
        super(QuestionEmbeddingModel, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
    def forward(self, sentence):
        
        tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.roberta_model(input_ids)
        
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]
        return cls_embedding

def main():
    embedder = QuestionEmbeddingModel()
    sentence = 'Who starred as lead actor in Harry Potter?'
    h_q = embedder(sentence)
    print(f'Question embedding size: {h_q.shape[1]}')
    
if __name__ == '__main__':
    main()