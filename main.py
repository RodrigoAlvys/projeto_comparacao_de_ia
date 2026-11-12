import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. CARREGAR E PREVER
# ==========================================
dataset_token = load_from_disk("./datasets/dataset_token")
dataset_token.set_format("torch")
modelo = BertForSequenceClassification.from_pretrained("./content/modelo_final")

trainer = Trainer(model=modelo)
resultados = trainer.predict(dataset_token)

# Probabilidades (Softmax) e previsões
logits = torch.tensor(resultados.predictions)
probabilidades = F.softmax(logits, dim=-1).numpy()
previsoes_finais = np.argmax(resultados.predictions, axis=-1)
gabarito = resultados.label_ids

# ==========================================
# 2. MÉTRICAS GERAIS
# ==========================================
acuracia = accuracy_score(gabarito, previsoes_finais)
print(f"\nAcurácia Geral: {acuracia:.2%}\n")

print("Relatório de Classificação:")
print(classification_report(
    gabarito, 
    previsoes_finais, 
    target_names=["0 (Negativo)", "1 (Neutro)", "2 (Positivo)"],
    zero_division=0
))

# ==========================================
# 3. DATAFRAME DE AUDITORIA
# ==========================================
# Carrega o JSONL como DataFrame (o arquivo tem 1 comentário por linha)
df_resultados = pd.read_json("./datasets/data_test.jsonl", lines=True, orient="records", encoding="utf-8")

# Adiciona as previsões
df_resultados['classe_esperada'] = df_resultados['label']
df_resultados['classe_prevista'] = previsoes_finais
df_resultados['acertou'] = (df_resultados['classe_esperada'] == df_resultados['classe_prevista']).astype(int)

# Adiciona as probabilidades
df_resultados['prob_negativo'] = probabilidades[:, 0]
df_resultados['prob_neutro'] = probabilidades[:, 1]
df_resultados['prob_positivo'] = probabilidades[:, 2]

# ==========================================
# 4. IMPRIMIR ERROS
# ==========================================
erros = df_resultados[df_resultados['acertou'] == 0]

print(f"\nO modelo errou {len(erros)} de {len(df_resultados)} comentários.")
print("\n--- Alguns exemplos de erros ---")
for index, row in erros.head(10).iterrows():
    print(f"Comentário: {row['comment']}")
    print(f"Real: {row['classe_esperada']} | Previsão: {row['classe_prevista']}\n")

# Salva o CSV final para abrir no Excel
df_resultados.to_csv("resultado_previsoes.csv", index=False)
print("Resultados salvos em 'resultado_previsoes.csv'")

# ==========================================
# 5. MATRIZ DE CONFUSÃO (PARA O ARTIGO)
# ==========================================
print("\nGerando Matriz de Confusão...")
cm = confusion_matrix(gabarito, previsoes_finais)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negativo (0)', 'Neutro (1)', 'Positivo (2)'],
            yticklabels=['Negativo (0)', 'Neutro (1)', 'Positivo (2)'])
plt.title('Matriz de Confusão - BERTimbau')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Classe Real (Gabarito)')
plt.tight_layout()
plt.savefig("matriz_confusao_bert.png")
print("Imagem 'matriz_confusao_bert.png' salva com sucesso!")
