##  Assessing Topic Modeling Techniques on Brazilian Public Bidding Data: A Comparative Study of LDA and BERTopic

### 1. **Reproduce Code**

#### 1.1. Setup Libraries

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Download spaCy language model:
    ```bash
    python -m spacy download pt_core_news_md
    ```

#### 1.2. Get the Data

In Linux:
1. Install `curl`:
    ```bash
    sudo apt install curl
    ```

2. Run the data retrieval script:
    ```bash
    python get_data_tce.py
    ```

Alternatively, download the data from [Google Drive](https://drive.google.com/file/d/1w9Y5qKA2sRa9PjwAedeRWDPmGmGnFdwc/view?usp=sharing) and place it in the `/data` directory.

#### 1.3. Installing CUML (If You Have a GPU)

When running the HDBSCAN, K-Means, and UMAP models, the `CUML` library is used. It's designed to run on GPUs, making computation much faster. If you don't have a GPU, the equivalent libraries from `sklearn` are commented in the code, so you can just uncomment them.

Check how to install `CUML` for your system [here](https://docs.rapids.ai/install#rapids-release-selector).

For my machine, the installation was:

```
pip install \
  --extra-index-url=https://pypi.nvidia.com \
  cudf-cu11==24.4.* cuml-cu11==24.4.*
```

Note: If you encounter this error `ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects`, execute:

 ```
 sudo apt-get install python3-dev
 ```

### 2. **About the Scripts**

#### 2.1. Load and Clean Data

- `get_data_tce.py`: Loads data from TCE, saving it to `data/tce_licitations.csv`.
  
- `data_cleaning.py`: Cleans the data.

#### 2.2. LDA

- `lda_opt.py`: Runs Optuna optimization for the LDA model.
  
- `lda_runs.py`: Runs the LDA model with the best parameters N times to average the results.
  
- `lda_fit.py`: Fits a single LDA model, if needed.
  
- `evaluate_lda.py`: Functions to evaluate the model.
  
- `research/lda_eval.ipynb`: Notebook to evaluate the model using `evaluate_lda.py`.

#### 2.3. BERTopic

- `bertopic_save_embeddings.py`: Saves the BERTopic embeddings, vectorizer model, and documents in pickle format. This is done to make optimization faster, so we don't have to get the embeddings every time when optimizing.
  
- `bertopic_opt.py`: Runs Optuna optimization for the BERTopic model.
  
- `bertopic_fit.py`: Fits a single BERTopic model, if needed (only available for HDBSCAN clustering model).
  
- `evaluate_bertopic.py`: Functions to evaluate the model.
  
- `research/bertopic_eval.ipynb`: Notebook to evaluate the model using `evaluate_bertopic.py`.
