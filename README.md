# :blossom: KNN-LM for LLaMa-GTPQ at scale

## :blossom: About
This repository is based on [neulab/knn-transformers](https://github.com/neulab/knn-transformers) repositoty and contains the implementation or [knn-lm](https://arxiv.org/abs/1911.00172).

For the information on how to use the code and on the theoretical background please contact the initial repository.

## :blossom: What is new
We extend [neulab/knn-transformers](https://github.com/neulab/knn-transformers) in several ways:
1. Support for quantized LlaMa models. Basically, if you need to run any other model, better do it with the initial repository.
2. Support for using large datasets in knn storage, namely saving faiss index in batches. This is done with ```build_index_on_the_go``` flag in knn arguments.
3. Support for using [semem](https://arxiv.org/abs/2303.01421) thresholds to memorize less data in the storage. Done with the help of ```semem_thres``` flag in knn arguments.
