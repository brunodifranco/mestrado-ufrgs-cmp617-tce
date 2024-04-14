# DAR UMA OLHADA EM https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#pre-compute-embeddings - Pre-compute embeddings¶



# mestrado-ufrgs-cmp617-tce

## Falta

Para a LLM:
- Ver a questao do vec (se da pra melhorar o tokenizer)
- Testar com outros modelos, e outras classes na zero shot list
- Testar as outras metricas de coherence tbm, e para o LDA tambem

Para o BertTopic (precisa fazer com batch size provavelmente)
- Testar com zero shot (VER PQ NAO ESTA INDO CERTO AS CLASSES (OLHAR NOTEBOOK DO BERTOPIC))

- Fazer funcionar a questão do nr_topics e zeroshot_topic_list


# para o LLM e BertTopic
- Testar primeiro somente com zero shot mesmo (se o resultado for melhor ja, tudo certo)
- Se o resultado for pior que o LDA ai sim fazer com fine tuning (no caso passar alguns exemplos pra rede)



Fazer os plots com PCA e UMAP, alem das word clouds


Depois disso tudo, só precisamos escrever o artigo


## Todo

- Separar cada etapa do processo em vários notebooks
- No final, com o código já pronto, ter scripts pra cada um poder reproduzir, como se fosse uma pipeline mesmo (ter um script main, q puxa varios - o main seria oq executa a pipeline).  E ter no readme oq cada script .py esta fazendo

## obs

instalacao spacy

python -m spacy download pt_core_news_md


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
