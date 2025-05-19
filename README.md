# KGQA: Question Answering with Knowledge Graphs

The code is designed to take a natural language question, process it using a Language Model (ROBERTa) to extract known entities, and then leverage knowledge graph embeddings (ComplEx) to find the most relevant entities that answer the question

## To run the code:
Create conda environment using the steps mentioned in requirements.txt

Download IMDb dataset, QA dataset and the test set, using the command 
    
    bash download.sh

To create triples, run 
    
    python preprocessing/preprocess_imdb.py

triples will be saved in `./kge/data/IMDb/`

Install kge, follow instructions mentioned [here](./kge/README.md)

To create KG Embeddings, 
    
    cd ./kge/data/
    
    python preprocess/preprocess_default.py IMDb

    cd ../../

    kge start train_config.yaml

To finetune RoBERTa for KGQA, 
In line 247, change `kge_checkpoint` to path of kge_embeddings saved
Make sure at line 250 `eval_only = False`

    python ./roberta/model.py

### Knowledge Graph Structure

The knowledge graph consists of the following:
* **Entities:** Title, Genre, Year, Actor, Director
* **Relations:** `Title hasGenre Genre`, `Title directedBy Director`, `Title starring Actor`, `Title releasedIn Year`, `Actor workedFor Director`, `Actor starredWith Co-actor`

### KGQA Module

1.  A question is processed by a Language Model (ROBERTa) to extract known entities (e.g., year, actors).
2.  These entities are then passed through Fully Connected (FC) layers.
3.  ComplEx embeddings are used to score and rank potential answer entities from the knowledge graph.
4.  The system returns the top-k entities as answers.

## Training

* The model is trained to minimize the negative log likelihood of the predicted answer distribution compared to the target distribution.
* Techniques like label smoothing and KL Divergence are used during training.
* The scoring function used is based on ComplEx embeddings.

## Evaluation

The system was evaluated using Mean Reciprocal Rank (MRR):
* MRR@1: 0.250
* MRR@5: 0.354
* MRR@10: 0.354
* Validation MRR: 0.38

**Capabilities:**
* Identifies the entity type of the answer (e.g., year, actor, title).
* Performs well with questions containing multiple entities.
* Capable of multi-hop answering by scoring all entities.

## Future Work

* Improving multi-hop answering by focusing on relevant subgraphs.
* Identifying answer entity-types more robustly.
* Utilizing text descriptions associated with entities (e.g., titles) to find relevant subgraphs.

## Examples

* **Question:** In which year did a thriller film release, featuring actors Jake Gyllenhaal and Rene Russo, with a title related to the world of art?
    * **Top-10 Predicted Answers:** `['2016', '2014', 'christian_bale', '2013', '2006', 'fantasy', '2009', 'mark_wahlberg', 'julianne_hough', 'cate_blanchett']`
    * **Correct Answer:** `['2014', 'nightcrawler']`

* **Question:** Name a 2016 movie in which Mark Ruffalo, Rachel McAdams, and Michael Keaton portrayed journalists investigating a real-life scandal.
    * **Top-10 Predicted Answers:** `['spotlight', 'ben_winchell', 'nick_frost', 'simon_pegg', 'maria_bello', 'avengers:_age_of_ultron', 'minions', 'tina_fey', 'mark_ruffalo', 'x-men_origins:_wolverine']`
    * **Correct Answer:** `['2015', 'spotlight']`
