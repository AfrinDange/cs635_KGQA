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
        id = 1
        integral_titles = [] # added column to dataset
        for title in dataset['Title'].to_list():
            if title not in title_ids:
                title_ids[title] = id
                integral_titles.append('title_'+str(id))
                id += 1
            else: 
                if title == 'The Host': # for the second "The Host" title 
                    title_ids[title+'_'] = id
                    integral_titles.append('title_'+str(id))
                    id += 1
                else:
                    integral_titles.append('title_'+str(title_ids[title]))
        dataset['title_id'] = integral_titles
        json.dump(dict(sorted(title_ids.items())), open('dataset/metadata/title_ids.json', 'w'), indent=4)
    
    if 'Genre' in entity_columns:
        '''
        Titles have one or more genres
        '''
        genre_ids = {} # saved in json
        id = 1
        integral_genres = []
        for genres in dataset['Genre'].to_list():
            title_genres = [] # genres assigned to the row/title
            for genre in genres.split(','):
                if genre not in genre_ids:
                    genre_ids[genre] = id
                    title_genres.append('genre_'+str(id))
                    id += 1
                else:
                    title_genres.append('genre_'+str(genre_ids[genre]))
            integral_genres.append(title_genres)
        dataset['genre_id'] = integral_genres
        json.dump(dict(sorted(genre_ids.items())), open('dataset/metadata/genre_ids.json', 'w'), indent=4)

    if 'Actors' in entity_columns:
        '''
        Titles have multiple actors
        '''
        actor_ids = {}
        id = 1
        integral_actors = []
        for actors in dataset['Actors'].to_list():
            title_actors = [] # actors from a title/row
            for actor in actors.split(','):
                actor = actor.strip() # trim whitespaces
                if actor not in actor_ids:
                    actor_ids[actor] = id
                    title_actors.append('actor_'+str(id))
                    id += 1
                else:
                    title_actors.append('actor_'+str(actor_ids[actor]))
            integral_actors.append(title_actors)
        dataset['actor_id'] = integral_actors
        json.dump(dict(sorted(actor_ids.items())), open('dataset/metadata.actor_ids.json', 'w'), indent=4)

    if 'Director' in entity_columns:
        '''
            Each Title has one director
        '''
        director_ids = {}
        id = 1
        integral_directors = []
        for director in dataset['Director'].to_list():
            if director not in director_ids:
                director_ids[director] = id
                integral_directors.append('director_'+str(id))
                id += 1
            else:
                integral_directors.append('director_'+str(director_ids[director]))
        dataset['director_id'] = integral_directors
        json.dump(dict(sorted(director_ids.items())), open('dataset/metadata/director_ids.json', 'w'), indent=4)

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

    os.makedirs('./complex/datasets/imdb/', exist_ok=True)
    save_split(train_triples, './complex/datasets/imdb/train.txt')
    save_split(valid_triples, './complex/datasets/imdb/valid.txt')
    save_split(test_triples, './complex/datasets/imdb/test.txt')
    