import requests
import os
os.makedirs('data', exist_ok=True)

base_url = 'https://groups.csail.mit.edu/sls/downloads/movie/'

files = [
    'engtest.bio',
    'engtrain.bio',
    'trivia10k13test.bio',
    'trivia10k13train.bio'
]

for file_name in files:
    file_url = base_url + file_name
    response = requests.get(file_url)
    
    if response.status_code == 200:
        with open('data/'+file_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download {file_name}")
