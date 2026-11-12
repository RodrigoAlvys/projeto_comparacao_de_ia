"""Demonstra o modelo salvo sem modificar o experimento."""
import argparse
from pathlib import Path
import joblib

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('texto')
p.add_argument('--modelo',type=Path,default=Path(__file__).resolve().parent/'resultados/modelo_tfidf_regressao.joblib')
a=p.parse_args()
if not a.texto.strip(): p.error('Digite um comentario nao vazio.')
modelo=joblib.load(a.modelo)
classes={0:'negativo',1:'neutro ou ambiguo',2:'positivo'}
prob=modelo.predict_proba([a.texto])[0]
rotulo=int(modelo.classes_[prob.argmax()])
print('Classe prevista:',classes[rotulo])
for c,v in zip(modelo.classes_,prob): print(f'{classes[int(c)]}: {v:.1%}')
print('Probabilidades estimadas pelo modelo; nao sao garantia de acerto.')
