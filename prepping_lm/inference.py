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
input_question ="In which year did a thriller film release, featuring actors Jake Gyllenhaal and Rene Russo, with a title related to the world of art?"
# tokenized_input = tokenizer.encode(
#     input_question,
#     return_tensors='pt'
# ).to('cuda:0')
tokenized_input = tokenizer.encode(
        "Answer the question: "+input_question,
        return_tensors='pt'
    ).to('cuda:0')

out = model.generate(
    tokenized_input,
    max_length=8,
    num_beams=2,
    early_stopping=True
)

# decode and print final answer
print('Answer: ', tokenizer.decode(out[0], skip_special_tokens=True))
