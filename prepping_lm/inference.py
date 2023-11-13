from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# checkpoint used for finetuning
checkpoint='t5-base'

tokenizer = T5Tokenizer.from_pretrained(checkpoint)

model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# load saved model and tranfer to GPU
model.load_state_dict(torch.load('./model_runs/test/finetuned_lm.pth').state_dict())
model.to('cuda:0')

# inference
input_question ="Which film stars Leonardo DiCaprio and was released in 2015?"
tokenized_input = tokenizer.encode(
    input_question,
    return_tensors='pt'
).to('cuda:0')

out = model.generate(
    tokenized_input,
    max_length=8,
    num_beams=2,
    early_stopping=True
)

# decode and print final answer
print('Answer: ', " ".join([tokenizer.decode(id) for id in out]))
