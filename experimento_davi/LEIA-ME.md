# Experimento de Davi: análise de sentimentos

Treinamento e avaliação executados: **76 acertos em 100 comentários, acurácia de 76% e F1 macro de 0,7420**. Não é necessário rodar novamente para consultar os resultados.

O modelo é TF-IDF + Regressão Logística, executado localmente. É uma alternativa ao Google Cloud, cujo cadastro não foi concluído. O grupo deve registrar essa substituição no artigo e comparar com os modelos que efetivamente executou.

## Abra primeiro

1. `resultados/RESULTADOS.md`: metodologia, números e limitações.
2. `resultados/matriz_confusao.png`: gráfico dos acertos e erros.
3. `resultados/previsoes_teste.csv`: as 100 previsões, com rótulos de referência.
4. `ROTEIRO_DAVI.md`: explicação curta para apresentar sua parte.

## Arquivos principais

- `executar_experimento.py`: limpeza, separação treino/validação, treinamento e avaliação.
- `preparar_teste.py`: conversor do teste agrupado para um comentário por linha.
- `prever.py`: demonstra o modelo com uma frase nova, sem retreinamento.
- `dados/dataset.csv`: CSV recebido do Rodrigo, incluído no ZIP para reprodução.
- `dados/teste_ia_original.jsonl`: teste comum preservado byte a byte.
- `resultados/modelo_tfidf_regressao.joblib`: modelo salvo. Carregar somente modelos de origem confiável.
- `resultados/protocolo.json`: regras, versões, hashes e quantidades.
- `resultados/particoes.csv`: quais registros do CSV foram para treino ou validação; a numeração começa em 1 após o cabeçalho e se refere a registros CSV, não linhas físicas.
- `resultados/linhas_removidas.json`: motivos das exclusões.

## Repetir no Windows

Use Python 3.12. Abra o terminal na pasta extraída e execute:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe executar_experimento.py --treino dados/dataset.csv --saida resultados_nova_execucao
```

A saída precisa ser uma pasta nova para preservar a execução anterior. Não escolha parâmetros observando os resultados nos 100 comentários. O experimento usa somente o comentário, sem nome ou descrição do produto; confirme esse procedimento com os colegas.

Para demonstrar com uma frase nova:

```powershell
.\.venv\Scripts\python.exe prever.py "O produto chegou quebrado e não funciona."
```

Essa demonstração não acrescenta exemplos à avaliação oficial.

## Dados e limites da conclusão

O CSV tinha 129.019 registros. Excluímos 2.353 repetições com mesmo rótulo e 50 registros correspondentes a 24 textos com rótulos conflitantes. Restaram 126.616 registros: 101.292 para treino e 25.324 para validação. Não houve sobreposição exata normalizada com o teste. Nenhum parâmetro foi escolhido pelos resultados do teste.

O usuário confirmou: CSV 0 negativo, 1 neutro ou ambíguo, 2 positivo. Eles correspondem a -1, 0 e 1 no teste. Essa definição mais ampla da classe intermediária deve constar no artigo. Nenhum rótulo original do teste foi alterado.

O teste tem 40 positivos, 43 negativos e 17 neutros. No grupo do livro, que inclui ironia, o modelo acertou 7 de 20. Isso ajuda a discutir limitações, mas não isola experimentalmente o efeito da ironia.

Confirmar com Rodrigo a origem do CSV, a regra de rotulagem e se o teste foi criado com auxílio de IA. Os arquivos não fornecem metadados suficientes para afirmar que combinam B2W e MQD. Antes de publicar dados no GitHub, documentar fonte e licença. O `.gitignore` exclui o CSV de treinamento e o modelo salvo por padrão. O ZIP é uma entrega local, não uma publicação no GitHub.

Para a comparação final, reunir os resultados dos outros integrantes. Estes arquivos descrevem somente este modelo.
