# Resultado de Davi: TF-IDF + Regressão Logística

Execução local real com Python/scikit-learn. Substitui o serviço Google Cloud no experimento; a troca deve ser registrada pelo grupo no artigo.

## Teste comum de 100 comentários

- Acertos: 76/100.
- Acurácia: 76.00%.
- F1 macro (três classes): 0.7420.
- Referência que sempre prevê a classe mais frequente do treino: 40.00% de acurácia.
- Referência que sempre responde negativo: 43.00% de acurácia.

## Como foi feito

Recebemos 129,019 registros em dataset.csv. Foram excluídas linhas vazias, duplicatas, todos os textos com rótulos conflitantes e sobreposições exatas normalizadas com o teste, quando presentes. Remoções nesta execução: {'duplicata': 2353, 'rotulos_conflitantes': 50}. Restaram 126,616 textos. Destinamos 101,292 ao treino e 25,324 à validação, por divisão estratificada 80/20 com semente 42. Os 100 comentários externos permaneceram separados. Os números das classes foram convertidos conforme confirmação do usuário: treino 0/1/2 corresponde a teste -1/0/1. A classe 1 do treino significa neutro ou ambíguo.

Usamos somente o comentário como entrada. TF-IDF com palavras e pares de palavras, min_df=2, máximo de 100.000 atributos e frequência sublinear. Regressão Logística com C=1, class_weight=balanced, solver=lbfgs, tol=0,0001 e máximo de 1.000 iterações. Não removemos palavras de negação nem aplicamos lista de stopwords. O vocabulário foi aprendido somente no treino. Parâmetros foram fixados previamente; não houve busca de parâmetros, ajuste por erros do teste nem retreinamento após a avaliação.

## Validação interna (não confundir com teste comum)

Acurácia: 80.40%. F1 macro: 0.7263.

## Limitações para discussão

O teste é pequeno: 40 positivos, 43 negativos e 17 neutros. O último grupo contém comentários irônicos sobre um livro e nenhuma referência positiva. Há textos mistos rotulados como neutros. Isso limita a generalização e deve ser considerado ao analisar os erros. A procedência exata do CSV e a geração dos exemplos do teste ainda precisam ser confirmadas com Rodrigo; não atribuir todos os registros a B2W/MQD sem essa confirmação. Rótulos do grupo foram preservados. A verificação de duplicatas cobre igualdade após normalização, não paráfrases. O modelo não representa o Google Cloud Natural Language. A comparação com os colegas exige que usem a mesma versão do teste e informem se incluem a descrição do produto como entrada.

## Arquivos

`metricas.json`: métricas completas, incluindo cada classe e grupo. `previsoes_teste.csv`: todas as previsões e acertos/erros. `matriz_confusao.png`: gráfico. `protocolo.json`: parâmetros, hashes e versões. `particoes.csv`: linha de origem e partição. `linhas_removidas.json`: auditoria das exclusões. `modelo_tfidf_regressao.joblib`: modelo treinado (carregar somente arquivos confiáveis).

## Fontes técnicas

- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
