import sys
sys.path.append('./')
from transformers import pipeline, BertTokenizer, BertForTokenClassification, BertConfig
from roberta.dataset import QADataset

saved_model_path = "./ner/saved_models/"

model_config = BertConfig.from_pretrained(saved_model_path)
model = BertForTokenClassification.from_pretrained(saved_model_path, config=model_config)
tokenizer = BertTokenizer.from_pretrained(saved_model_path)

token_classifier = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

dataset = QADataset('./dataset/QA.txt')

for i in range(len(dataset)):
    question = dataset[i]['question']
    entities = token_classifier(question)
    if len(entities) == 0:
        print(question)
