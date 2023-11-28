import pandas as pd
import numpy as np
import json
import os
from ampligraph.evaluation import train_test_split_no_unseen

def create_unique_ids(dataset, entity_columns):
    if 'Title' in entity_columns:
        '''
        Each row has a unique string title except
        'The Host' title which occurs twice (both different movies)
        '''
        os.makedirs('dataset/metadata/', exist_ok=True)
        title_ids = {} # saved in json
        str_identifiers = [] # added column to dataset
        for title in dataset['Title'].to_list():
            if title not in title_ids:
                processed_title = title.replace(" ", "_")
                title_ids[title] = processed_title
                str_identifiers.append(processed_title)
            else: 
                if title == 'The Host': # for the second "The Host" title 
                    processed_title = title.replace(" ", "_") + "_2"
                    title_ids[title+'_'] = processed_title
                    str_identifiers.append(processed_title)
                else:
                    str_identifiers.append(title_ids[title])

        dataset['title_id'] = str_identifiers
        json.dump(dict(sorted(title_ids.items())), open('dataset/metadata/title_ids.json', 'w'), indent=4, ensure_ascii=False)
    
    if 'Genre' in entity_columns:
        '''
        Titles have one or more genres
        '''
        genre_ids = {} # saved in json
        str_identifiers = []
        for genres in dataset['Genre'].to_list():
            title_genres = [] # genres assigned to the row/title
            for genre in genres.split(','):
                if genre not in genre_ids:
                    processed_genre = genre.strip().lower().replace(" ", "_")
                    genre_ids[genre] = processed_genre
                    title_genres.append(processed_genre)
                else:
                    title_genres.append(genre_ids[genre])
            str_identifiers.append(title_genres)
        dataset['genre_id'] = str_identifiers
        json.dump(dict(sorted(genre_ids.items())), open('dataset/metadata/genre_ids.json', 'w'), indent=4, ensure_ascii=False)

    if 'Actors' in entity_columns:
        '''
        Titles have multiple actors
        '''
        actor_ids = {}
        str_identifiers = []
        for actors in dataset['Actors'].to_list():
            title_actors = [] # actors from a title/row
            for actor in actors.split(','):
                actor = actor.strip().lower() # trim whitespaces
                if actor not in actor_ids:
                    processed_actor = actor.replace(" ", "_")
                    actor_ids[actor] = processed_actor
                    title_actors.append(processed_actor)
                else:
                    title_actors.append(actor_ids[actor])
            str_identifiers.append(title_actors)
        dataset['actor_id'] = str_identifiers
        json.dump(dict(sorted(actor_ids.items())), open('dataset/metadata/actor_ids.json', 'w'), indent=4, ensure_ascii=False)

    if 'Director' in entity_columns:
        '''
            Each Title has one director
        '''
        director_ids = {}
        str_identifiers = []
        for director in dataset['Director'].to_list():
            if director not in director_ids:
                processed_director = director.strip().lower().replace(" ", "_")
                director_ids[director] = processed_director
                str_identifiers.append(processed_director)
            else:
                str_identifiers.append(director_ids[director])
        dataset['director_id'] = str_identifiers
        json.dump(dict(sorted(director_ids.items())), open('dataset/metadata/director_ids.json', 'w'), indent=4, ensure_ascii=False)

    # Year is an int column

def create_triples(row):
    '''
        Title hasGenre Genre
        Title directedBy Director
        TItle starring Actor
        Title releasedIn Year
        Actor workedFor Director
        Actor starredWith Co-actor

        # ComplEx requires: subject_entity_id\trelation_id\tobject_entity_id
    '''
    triples = ''
    title = row['title_id']
    # hasGenre Relation
    for genre in row['genre_id']:
        triples += f'{title}\thasGenre\t{genre}\n'
    
    # directedBy Relation
    triples += f'{title}\tdirectedBy\t{row["director_id"]}\n'

    # starring Relation
    for i, actor in enumerate(row['actor_id']):
        triples += f'{title}\tstarring\t{actor}\n'
        triples += f'{actor}\tworkedFor\t{row["director_id"]}\n'
        for coactor in row['actor_id'][i+1:]:
            if actor != coactor:
                triples += f'{actor}\tstarredWith\t{coactor}\n'

    # releasedIn Relation
    triples += f'{title}\treleasedIn\t{row["Year"]}\n'
    
    return triples
    
def save_split(triples, path):
    '''
        Saves the train.txt, valid.txt, test.txt in ./complex/datasets/imdb/
    '''
    triples_string = '\n'.join(['\t'.join(triple) for triple in triples])
    with open(path, 'w') as f:
        f.write(triples_string)
    print(f'saved at {path}')

if __name__ == '__main__':
    data = pd.read_csv('dataset/IMDB.csv')
    
    # mention the entity columns
    entity_columns = ['Title', 'Genre', 'Director', 'Actors', 'Year']

    # process & create unique id for each string entities
    create_unique_ids(data, entity_columns)
    
    # numerical_entity_columns = [] # not adding year as a numerical entity as we have comparison based queries e.g. movie released in 2015 starring vin diesel 
    # process numerical entities

    # create triples
    # for each row returns all the triples in string format entity1\trelationA\tentity2\nentity3\trelationB\tentity4\n
    triples = '' 
    for i, row in data.iterrows():
        triples += create_triples(row)
    
    # save in .txt file
    with open('dataset/imdb_kg.txt', 'w') as f:
        f.write(triples)

    triples_list = np.array([triple.split('\t') for triple in triples.split('\n') if triple != ''])
    # Create train, valid, test split such that each entity or relation individually appearing in test/valid is also present in train
    # ComplEx performs link prediction
    train_valid_triples, test_triples = train_test_split_no_unseen(triples_list, test_size=0.1, seed=42)
    train_triples, valid_triples = train_test_split_no_unseen(train_valid_triples, test_size=test_triples.shape[0], seed=42)

    save_dir = './kge/data/IMDb/'
    os.makedirs(save_dir, exist_ok=True)

    # remove old files
    files = os.listdir(save_dir)
    for file in files:
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    save_split(train_triples, os.path.join('./kge/data/IMDb/', 'train.txt'))
    save_split(valid_triples, os.path.join('./kge/data/IMDb/', 'valid.txt'))
    save_split(test_triples, os.path.join('./kge/data/IMDb/', 'test.txt'))
    