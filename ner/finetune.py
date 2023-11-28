from datasets import DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification
import numpy as np
import evaluate

def compute_metrics(eval_preds):
    metric = evaluate.load('seqeval')
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]

    true_predictions = [
        [label_names[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    with open('predictions.txt', 'w') as file:
        for true_label, prediction in zip(true_labels, true_predictions):
            file.write(f"True label: {true_label}\n")
            file.write(f"Prediction: {prediction}\n\n")

    # Set zero_division=1 to replace undefined values with 1 in precision and F-score calculations
    all_metrics = metric.compute(
        predictions=true_predictions,
        references=true_labels,
        zero_division=1
    )

    return {
        "precision": all_metrics['overall_precision'],
        "recall": all_metrics['overall_recall'],
        "f1": all_metrics['overall_f1'],
        "accuracy": all_metrics['overall_accuracy']
    }


def align_labels_with_tokens(labels, word_ids):
  new_labels = []
  current_word=None
  for word_id in word_ids:
    if word_id != current_word:
      current_word = word_id
      label = -100 if word_id is None else labels[word_id]
      new_labels.append(label)

    elif word_id is None:
      new_labels.append(-100)

    else:
      label = labels[word_id]

      if label%2==1:
        label = label + 1
      new_labels.append(label)

  return new_labels

def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

  all_labels = examples['ner_tags']

  new_labels = []
  for i, labels in enumerate(all_labels):
    word_ids = tokenized_inputs.word_ids(i)
    new_labels.append(align_labels_with_tokens(labels, word_ids))

  tokenized_inputs['labels'] = new_labels

  return tokenized_inputs

if __name__ == '__main__':
    dataset = DatasetDict.load_from_disk('./ner/data/hf_data/movies_ner')

    model_checkpoint = 'bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    label_names = ner_feature = dataset['train'].features['ner_tags'].feature.names
    id2label = {i:label for i, label in enumerate(label_names)}
    label2id = {label:i for i, label in enumerate(label_names)}

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'ner_tags'])

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="./ner/saved_models/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model('./ner/saved_models/')
