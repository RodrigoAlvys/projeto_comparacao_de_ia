# Como explicar minha parte

TF-IDF transforma palavras e pares de palavras em números que representam sua importância nos textos. A Regressão Logística aprende relações entre esses números e as classes. Apesar de "regressão" no nome, aqui ela funciona como classificador.

## Fala sugerida

"Minha parte foi implementar TF-IDF com Regressão Logística como alternativa local, depois que não consegui concluir o cadastro do Google Cloud. Usei o CSV disponibilizado pelo grupo. Após retirar repetições e textos com rótulos conflitantes, separei 80% para treinamento e 20% para validação interna. O modelo recebeu somente o comentário.

Depois avaliei nos mesmos 100 comentários definidos pelo grupo. O modelo acertou 76, ou seja, teve 76% de acurácia. O F1 macro foi 0,742. Essa medida calcula o F1 de cada classe e faz a média, dando o mesmo peso às três classes.

Ele identificou corretamente 38 dos 40 positivos, 13 dos 17 neutros ou ambíguos e 25 dos 43 negativos. No grupo do livro, que inclui ironias, foram 7 acertos em 20. Isso mostra uma limitação nesses exemplos, sem provar que a ironia seja a única causa dos erros. A comparação com os outros modelos depende dos resultados dos colegas."

## Perguntas possíveis

**Treinou com o teste?** Não. Os 100 comentários ficaram separados. O vocabulário TF-IDF foi aprendido somente no treino.

**É o modelo do Google?** Não. É um classificador local com scikit-learn. A substituição precisa ser registrada no artigo.

**Por que usar esse método?** É uma referência tradicional de classificação de textos, executável sem serviço pago de nuvem, para comparar com as alternativas do grupo.

**Por que os números das classes diferem?** O CSV usa 0/1/2, enquanto o teste usa -1/0/1. A correspondência foi confirmada antes da avaliação.

**76% vale para qualquer texto?** Não. Esse resultado vale para os 100 comentários avaliados. Generalizar exige uma amostra maior e mais representativa.

**E os 80,40%?** É a acurácia da validação interna, feita com parte do CSV. Para comparar com os colegas, usar os 76% do teste comum, e não a validação interna.
