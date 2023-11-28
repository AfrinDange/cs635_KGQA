from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value

ner_labels = ['B-Actor', 'I-Actor', 'O', 'B-Plot', 'I-Plot', 'B-Opinion', 'I-Opinion', 'B-Award', 'I-Award', 'B-Year', 'B-Genre', 'B-Origin', 'I-Origin', 'B-Director', 'I-Director', 'I-Genre', 'I-Year', 'B-Soundtrack', 'I-Soundtrack', 'B-Relationship', 'I-Relationship', 'B-Character_Name', 'I-Character_Name', 'B-Quote', 'I-Quote']

def create_list_of_sequences(path):
    with open(path, 'r') as f:
        data = f.read().strip().split('\n\n')

    token_sequences, ner_tag_sequences = [], []
    for question in data:
        lines = question.split('\n')
        tokens, ner_tags = [], []

        for line in lines:
            ner_tag, token = line.split('\t')
            if ner_tag not in ner_labels: ner_labels.append(ner_tag)
            tokens.append(token)
            ner_tags.append(ner_tag)

        token_sequences.append(tokens)
        ner_tag_sequences.append(ner_tags)

    return {
        'tokens': token_sequences,
        'ner_tags': ner_tag_sequences
    }

features = Features({
    'tokens': Sequence(feature=Value('string')),
    'ner_tags': Sequence(feature=ClassLabel(names=ner_labels))
})

train_dataset = Dataset.from_dict(create_list_of_sequences('./ner/data/trivia10k13train.bio'), features=features)
test_dataset = Dataset.from_dict(create_list_of_sequences('./ner/data/trivia10k13test.bio'), features=features)

dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

dataset.save_to_disk('./ner/data/hf_data/movies_ner')
