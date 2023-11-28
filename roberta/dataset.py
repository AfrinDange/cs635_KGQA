from torch.utils.data import Dataset, DataLoader
import json
import re

def extract_year_from_question(question):
    pattern = r'\b\d{4}\b' 
    years_found = re.findall(pattern, question)
    return years_found[0] if years_found else ''

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

def build_actor_trie(actor_names):
    actor_trie = Trie()
    for actor_name in actor_names:
        actor_trie.insert(actor_name.lower()) 
    return actor_trie

def find_actors_in_sentence(actor_trie, sentence):
    found_actors = set()
    sentence = sentence.lower() 
    words = sentence.split()

    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            substr = ' '.join(words[i:j])
            if actor_trie.search(substr):
                found_actors.add(substr.replace(" ", "_"))

    return ",".join(found_actors)

class QADataset(Dataset):
    def __init__(self, path):
        with open('./dataset/metadata/actor_ids.json') as f:
            actors = list(json.load(f).keys())

        trie = build_actor_trie(actors)

        with open(path, 'r') as file:
            qa_text = file.read() 
        self.qa_pairs = []
        for pair in qa_text.split('\n\n'):
            if len(pair.split('\nAnswer: ')) == 1:
                print(pair)
            else:
                question, answer = pair.split('\nAnswer: ')
                instance = {
                    'question': question, 
                    'answer': answer, 
                    'actors': find_actors_in_sentence(trie, re.sub(r'[^\w\s\']', '', question)),
                    'year': extract_year_from_question(question)
                }
                self.qa_pairs.append(instance)

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        return self.qa_pairs[idx]
    
class QAEvalDataset(Dataset):
    def __init__(self, path):
        with open('./dataset/metadata/actor_ids.json') as f:
            actors = list(json.load(f).keys())

        trie = build_actor_trie(actors)

        with open(path, 'r') as file:
            qa_text = file.read() 
        self.qa_pairs = []
        questions, answers = qa_text.split('\n\n\n')
        questions = questions.split('\n\n')
        answers = [answer.split(" - ") for answer in answers.split("\n") if not answer.startswith("Answers")]

        for i in range(len(answers)):
            answers[i][1] = answers[i][1][1:-1].replace(" ", "_")

        for question, answer in zip(questions, answers):
            if question != "" and answer != "":
                instance = {
                    'question': question, 
                    'answer': answer, 
                    'actors': find_actors_in_sentence(trie, re.sub(r'[^\w\s\']', '', question)),
                    'year': extract_year_from_question(question)
                }
                self.qa_pairs.append(instance)

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        return self.qa_pairs[idx]


if __name__ == '__main__':

    dataset = QADataset('./dataset/QA.txt')

    count = 0
    for i in range(len(dataset)):
        if dataset[i]['actors'] == '' and dataset[i]['year'] == '':
            count += 1
            print(dataset[i]['question'], dataset[i]['actors'])
    print(f'No actors and year in {count} questions of {len(dataset)} questions')

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for batch in dataloader:
    #     print(batch) 
    #     break


    dataset = QAEvalDataset('./dataset/test.txt')

    count = 0
    for i in range(len(dataset)):
        if dataset[i]['actors'] == '' and dataset[i]['year'] == '':
            count += 1
            print(dataset[i]['question'], dataset[i]['actors'])
    print(f'No actors and year in {count} questions of {len(dataset)} questions')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        batch
    print(len(dataset))
        