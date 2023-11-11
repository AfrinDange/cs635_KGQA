### To run the code:
Download IMDb dataset, QA dataset and the test set, using the command `bash download.sh`

Make a `repos` directory in root to clone ComplEx, using the command `mkdir repos`

Clone the [ComplEx Repository](https://github.com/ttrouill/complex/tree/master), using the command

`git clone git@github.com:ttrouill/complex.git` or

`git clone https://github.com/ttrouill/complex.git`

To create triples, run `python preprocessing/preprocess_imdb.py`

To create KG Embeddings, 
run `cd repos/complex`

Modify the code in `fb15k_run.py`. For more details refer: [here](repos/complex/README.md).

run `python fb15k_run.py`