This repository contains code produced as part of the lecture Interdisciplinary Project in Data Science (2022W) at Vienna University of Technology. The project reconstructs historical valence-scores based on historical word-embeddings, using a regression model trained on contemporary valence-scores and contemporary word-embeddings. The suitability of these reconstructed valence-scores and the historical word-embeddings as an indicator for word-diversification is then investigated, with results suggesting that they are not suitable. More details can be found in the report *report.pdf*.


**Repository structure**
* run.sh: Calls the python files, executing all code needed for creating the output needed for the exploration notebook.
* explore.ipynb: Jupyter Notebook for exploring and visualizing results. Require the result-files generated either by run.sh or by running all the Jupyter Notebooks in \notebooks.
* notebooks: Contains Jupyter Notebooks which were used during development. They present the code in a commented and structured manner and can be used for visualization. Also contains evaluation.
* *.py files: Python files containing a subset of the code from the Jupyter Notebooks. They perform all the relevant operations for reconstructing historical valence-scores and finding the closest neighbors in a given historical embedding-space for each word. However, they do not contain code for evaluation.
* results_k_{}.pkl: result file generated at the end of code execution for a given neighborhood size. Required for running explore.ipynb.
* resources: Contains external resources such as pretrained word-embeddings or sentiment lexicons as well as internal resources such as intermediate results or aligned embeddings.
* representations, vecanalysis, googlengram: Code for embedding-space alignment, copied and uncleanly adapted for Python 3 from https://github.com/williamleif/histwords. 
* figures: Contains figures used for the report


Due to filesize limits, word-embeddings need to be download from their original sources:
* Historical word-embeddings (filenames: 1800-vocab.pkl, 1800-w.npy, 1810-vocab.pkl, ...) : https://nlp.stanford.edu/projects/histwords/
* Contemporary Geometry of Culture word-embeddings (filename: US_Ngrams_2000_12.csv): https://knowledgelab.github.io/GeometryofCulture/
* Contemporary Wikipedia2Vec word-embeddigns (filename: enwiki_20180420_300d.txt) :https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
All mentioned files need to be placed to /resources/word_embeddings/