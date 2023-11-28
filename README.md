### To run the code:
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
