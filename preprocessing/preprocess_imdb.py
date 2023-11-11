import pandas as pd
import json
import os
import random

def create_unique_ids(dataset, entity_columns):
    if 'Title' in entity_columns:
        '''
        Each row has a unique string title except
        'The Host' title which occurs twice (both different movies)
        '''
        os.makedirs('../dataset/metadata/', exist_ok=True)
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
        json.dump(dict(sorted(title_ids.items())), open('../dataset/metadata/title_ids.json', 'w'), indent=4)
    
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
        json.dump(dict(sorted(genre_ids.items())), open('../dataset/metadata/genre_ids.json', 'w'), indent=4)

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
        json.dump(dict(sorted(actor_ids.items())), open('../dataset/metadata.actor_ids.json', 'w'), indent=4)

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
        json.dump(dict(sorted(director_ids.items())), open('../dataset/metadata/director_ids.json', 'w'), indent=4)

    # Year is an int column

def create_tuples(row):
    '''
        Title hasGenre Genre
        Title directedBy Director
        TItle starring Actor
        Title releasedIn Year

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
    for actor in row['actor_id']:
        triples += f'{title}\tstarring\t{actor}\n'

    # releasedIn Relation
    triples += f'{title}\treleasedIn\t{row["Year"]}\n'

    return triples
    


if __name__ == '__main__':
    data = pd.read_csv('../dataset/IMDB.csv')
    
    # mention the entity columns
    entity_columns = ['Title', 'Genre', 'Director', 'Actors', 'Year']
    # process & create unique id for each string entities
    create_unique_ids(data, entity_columns)
    
    # numerical_entity_columns = [] # not adding year as a numerical entity as we have comparison based queries e.g. movie released in 2015 starring vin diesel 
    # process numerical entities

    # create tuples
    triples = ''
    for i, row in data.iterrows():
        triples += create_tuples(row)
    
    # save in .txt file
    with open('../dataset/imdb_kg.txt', 'w') as f:
        f.write(triples)

    # split into train:val:test 8:1:1
    title_ids = json.load(open('../dataset/metadata/title_ids.json', 'r'))
    title_ids = list(map(lambda x: 'title_' + str(x), title_ids.values()))

    num_titles = len(title_ids)

    random.seed(2023)

    random.shuffle(title_ids)
    # total titles = 1000
    train_titles = title_ids[:800] 
    valid_titles = title_ids[800:900]
    test_titles = title_ids[900:]
    print(f'Split {num_titles} into {len(train_titles)}:{len(valid_titles)}:{len(test_titles)}')

    train_triples=''
    valid_triples=''
    test_triples=''

    for triple in triples.split('\n'):
        if triple == '':
            continue
        title = triple.split('\t')[0]
        if title in train_titles:
            train_triples += triple + '\n'
        elif title in valid_titles:
            valid_triples += triple + '\n'
        elif title in test_titles:
            test_triples += triple + '\n'
        else:
            ValueError(f'{title} not in any split!')

    assert train_triples != '' and valid_triples != '' and test_triples != ''
    
    os.makedirs('../repos/complex/datasets/imdb/', exist_ok=True)
    with open('../repos/complex/datasets/imdb/train.txt', 'w') as f:
        f.write(train_triples)
    with open('../repos/complex/datasets/imdb/valid.txt', 'w') as f:
        f.write(valid_triples)
    with open('../repos/complex/datasets/imdb/test.txt', 'w') as f:
        f.write(test_triples)


    