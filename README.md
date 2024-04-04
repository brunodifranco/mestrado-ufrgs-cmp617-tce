# mestrado-ufrgs-cmp617-tce

## Falta

Para o BertTopic:
- Rodar em todo conjunto de dados
- Fazer funcionar a questão do nr_topics e zeroshot_topic_list


Para a LLM:
- Ver se vou fazer LLM mesmo, ou somente BertTopic
- Se sim, testar com zero shoot e testar o cálculo do Coherence Score da mesma forma que foi feita com BertTopic


Fazer os plots com PCA e UMAP


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
