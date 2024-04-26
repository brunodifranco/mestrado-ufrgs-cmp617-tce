
# mestrado-ufrgs-cmp617-tce

## Falta


Fazer os plots com PCA e UMAP, alem das word clouds


Depois disso tudo, só precisamos escrever o artigo


## obs

instalacao spacy

python -m spacy download pt_core_news_md

## Sobre os dados

foram separados 2021 a 2024, e somente aquelas licitacoes aprovadas


## IMPORTANTE MENCIONAR 

"But by its very nature, LDA is a generative probabilistic method. Simplifying a little bit here, each time you use it, many Dirichlet distributions are generated, followed by inference steps"

https://stackoverflow.com/questions/51956153/gensim-lda-coherence-values-not-reproducible-between-runs

## Pra ter acesso aos dados pre tratados:

No linux:
- Precisa ter o curl instalado `sudo apt install curl`
- Executar o script `get_data_tce.py`

Ou (em qualquer OS):

Tem no [google drive](https://drive.google.com/file/d/1w9Y5qKA2sRa9PjwAedeRWDPmGmGnFdwc/view?usp=sharing)

e colocar na pasta /data


## NA INSTALACAO DO CUML

So escrever assim "checkout how to install for your system in https://docs.rapids.ai/install#rapids-release-selector"

no meu caso foi 

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.4.* cuml-cu11==24.4.*
    

outra coisa, se tiver problema com bulding `ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects` precisa rodar `sudo apt-get install python3-dev`

## Obs

Devemos pegar o modelo e rodar x vezes, pra termos uma média +- std do modelo, justamente pela coherence ser um pouco variável

https://stackoverflow.com/questions/51956153/gensim-lda-coherence-values-not-reproducible-between-runs


## POR QUE FOI UTILIZADO CV pra otimizar no optuna?

"Our analyses demonstrate that the metrics CV and CP are more sensitive to noise. That confirms their applicability in scenarios where the user wants to highlight topics with some unrelated words", conforme o artigo https://sol.sbc.org.br/journals/index.php/jidm/article/download/2181/2049/11581

Por isso, será o escolhido.

