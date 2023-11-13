import numpy as np
import torch
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import argparse
import os

'''
Ref:
https://huggingface.co/docs/transformers/model_doc/t5 
https://huggingface.co/transformers/v3.0.2/model_doc/t5.html 
'''

parser = argparse.ArgumentParser(prog='Finetune LM for QA task')
parser.add_argument('--model_checkpoint', type=str, default='t5-base')
parser.add_argument('--num_epochs', type=int, default=16)
parser.add_argument('--bsz', type=int, default=16)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)

def set_seed(seed):
    '''
        set seed for reproducibility
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda_is_available():
    torch.cuda.manual_seed_all(seed)


def process_qa_pairs(path):
    '''
        reads the QA.txt file to create question answer pairs

        creates a tuple: (done as per the reference)
        ("Answer the question: <question> </s>", "<answer> </s>")
    '''
    with open(path, 'r') as f:
        qa_pairs = []
        question = ''
        for line in f.readlines():
            if line.startswith('Answer:'):
                qa_pairs.append(("Answer the question: "+question + " </s>", line[8:].strip()+" </s>"))
            else:
                question = line.strip()
    return qa_pairs

class QADataset(torch.utils.data.Dataset):
    '''
        Returns the string question and corresponding answer
    '''
    def __init__(self, data_path):
        qa_pairs = process_qa_pairs(data_path)
        self.questions, self.answers = zip(*qa_pairs)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]
    
    def __len__(self):
        return len(self.questions)
    
class Collate():
    '''
        Given a batch of input questions and output answers,
        performs tokenization in batched fashion
    '''
    def __init__(self, model_checkpoint):
        self.tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    def __call__(self, batch):
        '''
            The token ids are processed according to the reference
        '''
        questions, answers = zip(*batch)
        encoded_questions = self.tokenizer.batch_encode_plus(
            questions,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        encoded_answers = self.tokenizer.batch_encode_plus(
            answers,
            max_length=16,
            padding='max_length',
            return_tensors='pt'
        ).input_ids
        encoded_answers[encoded_answers == self.tokenizer.pad_token_id] = -100
        return encoded_questions, encoded_answers

if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(args.seed)

    ## create experiment directory
    save_dest = os.path.join('./model_runs', args.exp_name)
    if os.path.exists(save_dest):
        print('Experiment already exists')
        exit()
    else:
        os.makedirs(save_dest)
    
    # create dataloader
    dataset = QADataset('dataset/QA.txt')
    collate = Collate(args.model_checkpoint)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bsz, collate_fn=collate)

    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    epoch_loss = []
    batch_loss = []

    num_batches = len(dataloader)

    for epoch in range(args.num_epochs):
        running_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # forward
            x, y = batch
            output = model(
                input_ids=x['input_ids'].to(args.device), 
                attention_mask=x['attention_mask'].to(args.device),
                labels=y.to(args.device),
                # decoder_input_ids=y['input_ids'].to(args.device),
                # decoder_attention_mask=y['attention_mask'].to(args.device)
            )

            loss = output.loss.item()
            running_loss += loss
            batch_loss.append(loss)
            
            # backprop
            output.loss.backward()
            optimizer.step()

            # logging
            with open(os.path.join(save_dest, 'losses.txt'), 'a') as f:
                if i == num_batches - 1:
                    f.write(f'epoch={epoch}, current_loss={loss}, epoch_loss={running_loss/num_batches}\n')
                else:
                    f.write(f'epoch={epoch}, current_loss={loss}\n')
        epoch_loss.append(running_loss)

    # example inference
    input_question ="Which film stars Leonardo DiCaprio and was released in 2015?"
    tokenized_input = collate.tokenizer.encode(
        "Answer the question: "+input_question,
        return_tensors='pt'
    ).to('cuda:0')

    out = model.generate(tokenized_input)
    print('Answer: ', " ".join([collate.tokenizer.decode(id) for id in out]))

    # save model
    torch.save(model.to('cpu'), os.path.join(save_dest, 'finetuned_lm.pth'))

