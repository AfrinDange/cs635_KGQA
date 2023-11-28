import sys
sys.path.append('./')
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from roberta.dataset import QADataset, QAEvalDataset
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
import os

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KGQAModule(nn.Module):
    def __init__(self, config):
        
        super(KGQAModule, self).__init__()

        self.roberta_model = RobertaModel.from_pretrained(config['checkpoint'])
        self.device = config['device']

        self.tail_embedding = nn.Embedding.from_pretrained(config['kge'].get_o_embedder().embed_all().to(self.device), freeze=True)

        self.label_smoothing = config['label_smoothing']

        roberta_hidden_dim = 768
        self.linear = nn.Sequential(
            nn.Linear(roberta_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self._klloss = torch.nn.KLDivLoss(reduction='batchmean')

    def loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def complex_scores(self, head, relation):
        '''
            head: complex embedding of entity in question
            relation: embedding generated using roberta
        '''
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1) 
   
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.tail_embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        pred = score
        return pred

    def forward(self, batch):
        '''
            batch:
                questions:
                    input_ids:
                    attention_mask: 
                actors: 

                years: 

        '''
        # get question embedding from RoBERTa
        outputs = self.roberta_model(**batch['question'])
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        # project CLS embedding
        question_embedding = self.linear(cls_embedding)

        # head embedding
        head_embedding = batch['actors'] + batch['year']

        prediction = self.complex_scores(head_embedding, question_embedding)

        target = batch['answer']
        if self.label_smoothing > 0:
            target = ((1.0-self.label_smoothing)*target) + (1.0/target.size(1)) 
        loss = self.loss(prediction, target)

        return loss
    
    def get_scores(self, question, actors, year):
        outputs = self.roberta_model(**question)
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :]

        # project CLS embedding
        question_embedding = self.linear(cls_embedding)

        # head embedding
        head_embedding = actors + year

        prediction = self.complex_scores(head_embedding, question_embedding)

        prediction = torch.argsort(prediction, dim=1, descending=True)

        return prediction

class Collate():
    def __init__(self, tokenizer, kge, device='cuda:1'):
        self.tokenizer = tokenizer
        self.kge = kge
        self.device = device
        self.index = {k.lower(): v for v, k in enumerate(kge.dataset.entity_strings())}

    def get_actors_embedding(self, actors):
        if actors == '':
            return torch.zeros((1, 32)).to(self.device)
        else:
            actors = actors.split(',')
        idx = torch.Tensor([self.index[key] for key in actors if key in self.index]).long().to(self.device)
        if idx.numel() == 0:
            return torch.zeros((1, 32)).to(self.device)
        return torch.sum(self.kge.get_s_embedder().embed(idx), dim=0, keepdim=True) 
    
    def get_year_embedding(self, year):
        if year == '':
            return torch.zeros((1, 32)).to(self.device)
        else:
            if year in self.index:
                idx = self.index[year]
            else:
                return torch.zeros((1, 32))
            return self.kge.get_s_embedder().embed(torch.Tensor([idx]).long().to(self.device))
        
    def get_ohe_answers(self, answers):
        emb_len = len(self.index)
        answer_emb = []
        for answer in answers:
            emb = torch.zeros((1, emb_len), dtype=torch.float32)
            emb[:, answer] = 1
            answer_emb.append(emb)
        return torch.cat(answer_emb, dim=0)
        
    def __call__(self, batch):
        questions = self.tokenizer.batch_encode_plus(
            [b['question'] for b in batch],
            padding=True,
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        questions['input_ids'] = questions['input_ids'].to(self.device)
        questions['attention_mask'] = questions['attention_mask'].to(self.device)

        # add entity embeddings
        actors = torch.cat(list(map(lambda x: self.get_actors_embedding(x['actors']), batch)), dim=0).to(self.device)
        year = torch.cat(list(map(lambda x: self.get_year_embedding(x['year']), batch)), dim=0).to(self.device)

        if not type(batch[0]['answer']) == list: # train
            answers = self.get_ohe_answers(list(map(lambda x: self.index[x['answer'].lower().strip().replace(' ', '_')], batch))).to(self.device)
        else: # eval
            answers = [b['answer'] for b in batch]

        assert torch.isnan(actors).any() == False
        assert torch.isnan(year).any() == False

        

        return {
            'question': questions,
            'actors': actors,
            'year': year,
            'answer': answers
        }
    
def calculate_mrr_at_k(topk_results, correct_answers, k):
    '''
        topk_results: list of list, [i: example][j: jth result]
        correct:answer: list of list, [i: example][0: year, 1: movie_name]
    '''
    mrr_sum = 0
    num_queries = len(topk_results)

    for i in range(num_queries):
        year = correct_answers[i][0]
        answer = correct_answers[i][1]
        top_k = topk_results[i][:k]
        if  year in top_k:
            rank = top_k.index(year) + 1
            mrr_sum += 1 / rank
        if answer in top_k:
            rank = top_k.index(answer) + 1
            mrr_sum += 1 / rank

    mrr = mrr_sum / num_queries if num_queries > 0 else 0
    return mrr

if __name__ == '__main__':
    device = 'cuda:1'
    num_epochs = 100
    kge_checkpoint = './kge/local/experiments/20231128-140040-train_config/checkpoint_best.pt'
    set_seed(42)

    eval_only = False

    save_dir = './roberta/results/'
    exp_name = 'test_wo_kld_batch_mean'

    os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)

    checkpoint = load_checkpoint(kge_checkpoint)
    kge_model = KgeModel.create_from(checkpoint).to(device)

    config = {
        'checkpoint': 'roberta-base',
        'kge': kge_model,
        'device': device,
        'label_smoothing': 0.1
    }

    tokenizer = RobertaTokenizer.from_pretrained(config['checkpoint'])
    collate = Collate(tokenizer, kge_model, device=device)
    model = KGQAModule(config)
    model.to(device)
    
    if not eval_only: 
        dataset = QADataset('./dataset/QA.txt')
           
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        log_file = open(os.path.join(save_dir, exp_name, 'train_log.txt'), 'w')

        for epoch in tqdm(range(num_epochs)):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            log_file.write(f"Epoch {epoch+1}, Total Loss: {total_loss}\n")
            print(f"Epoch {epoch+1}, Total Loss: {total_loss}")
        log_file.close()
        torch.save(model.state_dict(), os.path.join(save_dir, exp_name, "model_final.pt"))
    else:
        model.load_state_dict(torch.load(os.path.join(save_dir, exp_name, "model_final.pt")))

    dataset = QAEvalDataset('./dataset/test.txt')  
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate) # keep bsz=1

    answers = []
    eval_predictions = []
    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            predictions = model.get_scores(batch['question'], batch['actors'], batch['year'])
            top_100 = list(kge_model.dataset.entity_strings(predictions.squeeze()[:100].long()))
            eval_predictions.append(top_100)
            answers.extend(batch['answer'])
        
        mrr1 = calculate_mrr_at_k(eval_predictions, answers, 1)
        mrr5 = calculate_mrr_at_k(eval_predictions, answers, 5)
        mrr10 = calculate_mrr_at_k(eval_predictions, answers, 10)
        mrr50 = calculate_mrr_at_k(eval_predictions, answers, 50)

        print("MRR@1: ", mrr1)
        print("MRR@5: ", mrr5)
        print("MRR@10: ", mrr10)
        print("MRR@50: ", mrr50)

        with open(os.path.join(save_dir, exp_name, 'results.txt'), 'w') as f:
            f.write(f'MRR@1 = {mrr1}\nMRR@5 = {mrr5}\nMRR@10 = {mrr10}\nMRR@50 = {mrr50}')

    
           
