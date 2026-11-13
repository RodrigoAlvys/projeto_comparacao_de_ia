# Experimento comparando TF-IDF e BERTimbau em analise de comentários
>[!NOTE]
>Esse é um projeto sobre IA valendo nota da segunda unidade da matéria de inteligência artificial que tem como objetivo.
>Esse repositório possui o **[artigo](documento/Artigo_comparacao_ia.pdf)** que sera avaliado e explica o projeto em detalhes.

## Requisitos
[requisitos](requirements.txt) do modelo BERTimbau:
* python 3.14+
* accelerate 1.15+
* datasets 5.2+
* matplotlib 3.11.2+
* numpy 2.5.4+
* pandas 3.1.1+
* transformers 5.22.2+
* torch 2.15+

## Metodologia
Foi utilizado o arquivo [dataset.csv](datasets/dataset.csv) que é um derivado do *B2W[^1]* criado para treinar os modelos. No arquivo original, existe uma lista de produtos com nome do produto,
comentário e nota de 1 a 5 nas quais foram mantido no dataset final com exceção da notas que foi convertido para label, onde valiações com 1 e 2 duas estrelas se tornou no número 0, mostrando que
o comentário é negativo, 3 se torna 1, representando comentários neutros e ambíguos, enquanto que 4 e e foram considerados positivos.

[^1]: *[B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01)* is an open corpus of product reviews. It contains more than 130k e-commerce customer reviews, 
collected from the Americanas.com website between January and May, 2018. B2W-Reviews01 offers rich information about the reviewer profile, such as gender, age, and geographical location.
