"""Experimento reproduzivel: TF-IDF + regressao logistica, sem servicos de nuvem."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import csv
import hashlib
import json
import platform
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
CLASSES=['negativo', 'neutro_ou_ambiguo', 'positivo']
MAPA_TESTE={-1:0, 0:1, 1:2}  # Correspondencia confirmada pelo usuario em 11/11/2026.

def normalizar(texto):
    return ' '.join(unicodedata.normalize('NFKC',texto).casefold().split())

def salvar_json(path,valor):
    path.write_text(json.dumps(valor,ensure_ascii=False,indent=2),encoding='utf-8')

def medir(y,pred):
    return {'n':len(y),'acertos':int(np.sum(np.asarray(y)==np.asarray(pred))),
            'acuracia':float(accuracy_score(y,pred)),
            'f1_macro':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'ordem_classes':CLASSES,
            'matriz_confusao':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'por_classe':classification_report(y,pred,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)}

def executar(dataset,teste,out):
    if (out/'metricas.json').exists():
        raise ValueError('Esta pasta ja tem resultados. Use outra --saida para preservar a execucao anterior.')
    out.mkdir(parents=True,exist_ok=True)
    comeco=time.perf_counter()
    registros=[]
    for n,linha in enumerate(teste.read_text(encoding='utf-8-sig').splitlines(),1):
        if not linha.strip(): continue
        grupo=json.loads(linha)
        for i,(texto,label) in enumerate(grupo['comentarios'],1):
            if not texto.strip() or type(label) is not int or label not in MAPA_TESTE:
                raise ValueError(f'Teste invalido em {n}/{i}')
            registros.append({'id':f'g{n:02d}_c{i:02d}','grupo':n,'texto':texto,'rotulo_original':label,'rotulo':MAPA_TESTE[label]})
    if not registros: raise ValueError('Teste vazio')
    chaves_teste={normalizar(r['texto']) for r in registros}
    if len(chaves_teste)!=len(registros): raise ValueError('Teste com textos repetidos; revisar protocolo.')
    with dataset.open(encoding='utf-8-sig',newline='') as f:
        leitor=csv.DictReader(f)
        if not {'review_text','label'}.issubset(leitor.fieldnames or []):
            raise ValueError('O treino precisa das colunas review_text e label.')
        rows=list(leitor)
    grupos=defaultdict(list)
    removidos=[]
    for i,r in enumerate(rows,1):
        texto=r['review_text'] or ''
        if r['label'] not in {'0','1','2'}: raise ValueError(f'Label desconhecido na linha {i}')
        if not texto.strip():
            removidos.append({'linha_dados':i,'motivo':'texto_vazio'})
            continue
        grupos[normalizar(texto)].append((i,texto,int(r['label'])))
    dados=[]
    for chave,itens in grupos.items():
        if chave in chaves_teste:
            removidos.extend({'linha_dados':i,'motivo':'sobreposicao_teste'} for i,t,y in itens)
        elif len({y for i,t,y in itens})>1:
            removidos.extend({'linha_dados':i,'motivo':'rotulos_conflitantes'} for i,t,y in itens)
        else:
            dados.append(itens[0])
            removidos.extend({'linha_dados':i,'motivo':'duplicata'} for i,t,y in itens[1:])
    if set(y for i,t,y in dados)!={0,1,2}: raise ValueError('Faltam classes no treino limpo')
    ids=np.arange(len(dados))
    tr,va=train_test_split(ids,test_size=.2,random_state=42,stratify=[d[2] for d in dados])
    Xtr=[dados[i][1] for i in tr]; ytr=[dados[i][2] for i in tr]
    Xva=[dados[i][1] for i in va]; yva=[dados[i][2] for i in va]
    assert not ({normalizar(t) for t in Xtr}&{normalizar(t) for t in Xva})
    assert not ({normalizar(t) for t in Xtr+Xva}&chaves_teste)
    motivos=dict(Counter(r['motivo'] for r in removidos))
    protocolo={'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
              'teste_sha256':hashlib.sha256(teste.read_bytes()).hexdigest(),
              'linhas_originais':len(rows),'contagem_original':dict(Counter(r['label'] for r in rows)),
              'remocoes':motivos,'textos_conflitantes':sum(len({y for i,t,y in v})>1 for v in grupos.values()),
              'linhas_limpas':len(dados),'n_treino':len(tr),'n_validacao':len(va),'n_teste':len(registros),
              'classes_treino':dict(Counter(ytr)),'classes_validacao':dict(Counter(yva)),
              'classes_teste':dict(Counter(r['rotulo'] for r in registros)),
              'semente':42,'entrada_modelo':'somente review_text/comentario; sem nome ou descricao do produto',
              'classes_csv':dict(enumerate(CLASSES)),'mapa_teste_para_csv':MAPA_TESTE,
              'origem_mapeamento':'Confirmacao do usuario: 0 negativo, 1 neutro ou ambiguo, 2 positivo.',
              'normalizacao_deduplicacao':'Unicode NFKC, casefold, normalizacao de espacos; nao remove pontuacao',
              'hiperparametros':'Fixados antes da avaliacao; sem busca ou selecao pelo teste.',
              'tfidf':{'ngram_range':[1,2],'min_df':2,'max_features':100000,'sublinear_tf':True,'lowercase':True,'stop_words':None},
              'regressao_logistica':{'C':1.0,'class_weight':'balanced','solver':'lbfgs','max_iter':1000,'tol':1e-4},
              'versoes':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'joblib':joblib.__version__}}
    salvar_json(out/'protocolo.json',protocolo)
    salvar_json(out/'linhas_removidas.json',removidos)
    with (out/'particoes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['linha_dados_original','particao','label'])
        for nome,indices in [('treino',tr),('validacao',va)]:
            w.writerows((dados[i][0],nome,dados[i][2]) for i in indices)
    modelo=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)),
                     ('classificador',LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=1000,tol=1e-4,random_state=42))])
    print(f'Treino: {len(tr)}; validacao: {len(va)}; teste: {len(registros)}',flush=True)
    inicio=time.perf_counter()
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always',ConvergenceWarning)
        modelo.fit(Xtr,ytr)
        treino_seg=time.perf_counter()-inicio
        predva=modelo.predict(Xva)
        inicio=time.perf_counter()
        probs=modelo.predict_proba([r['texto'] for r in registros])
        inferencia_seg=time.perf_counter()-inicio
    if any(issubclass(a.category,ConvergenceWarning) for a in avisos):
        raise RuntimeError('Treinamento nao convergiu; revisar antes de entregar resultados.')
    pred=modelo.classes_[np.argmax(probs,axis=1)]
    yt=[r['rotulo'] for r in registros]
    majoritaria=Counter(ytr).most_common(1)[0][0]
    metricas={'validacao_interna':medir(yva,predva),'teste_comum':medir(yt,pred),
              'referencia_classe_majoritaria_treino':medir(yt,[majoritaria]*len(yt)),
              'referencia_sempre_negativo':medir(yt,[0]*len(yt)),
              'por_grupo':{},'tempo_treino_segundos':treino_seg,'tempo_inferencia_teste_segundos':inferencia_seg,
              'tempo_total_segundos':time.perf_counter()-comeco,'iteracoes':modelo['classificador'].n_iter_.tolist(),
              'vocabulario':len(modelo['tfidf'].vocabulary_)}
    for g in sorted({r['grupo'] for r in registros}):
        indices=[i for i,r in enumerate(registros) if r['grupo']==g]
        metricas['por_grupo'][g]=medir([yt[i] for i in indices],[pred[i] for i in indices])
    salvar_json(out/'metricas.json',metricas)
    with (out/'previsoes_teste.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['id','grupo','texto','rotulo_teste_original','classe_esperada','classe_prevista','acertou']+['prob_'+c for c in CLASSES])
        for r,y,pr in zip(registros,pred,probs):
            w.writerow([r['id'],r['grupo'],r['texto'],r['rotulo_original'],CLASSES[r['rotulo']],CLASSES[y],int(y==r['rotulo'])]+[float(pr[list(modelo.classes_).index(c)]) for c in [0,1,2]])
    joblib.dump(modelo,out/'modelo_tfidf_regressao.joblib',compress=3)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(7,5.6))
    ConfusionMatrixDisplay(np.array(metricas['teste_comum']['matriz_confusao']),display_labels=['Negativo','Neutro/ambíguo','Positivo']).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set(xlabel='Classe prevista',ylabel='Classe de referência',title='TF-IDF + Regressão Logística\n100 comentários do teste comum')
    fig.tight_layout();fig.savefig(out/'matriz_confusao.png',dpi=180);plt.close(fig)
    resumo=f'''# Resultado de Davi: TF-IDF + Regressão Logística

Execução local real com Python/scikit-learn. Substitui o serviço Google Cloud no experimento; a troca deve ser registrada pelo grupo no artigo.

## Teste comum de 100 comentários

- Acertos: {metricas['teste_comum']['acertos']}/{len(yt)}.
- Acurácia: {metricas['teste_comum']['acuracia']:.2%}.
- F1 macro (três classes): {metricas['teste_comum']['f1_macro']:.4f}.
- Referência que sempre prevê a classe mais frequente do treino: {metricas['referencia_classe_majoritaria_treino']['acuracia']:.2%} de acurácia.
- Referência que sempre responde negativo: {metricas['referencia_sempre_negativo']['acuracia']:.2%} de acurácia.

## Como foi feito

Recebemos {len(rows):,} registros em dataset.csv. Foram excluídas linhas vazias, duplicatas, todos os textos com rótulos conflitantes e sobreposições exatas normalizadas com o teste, quando presentes. Remoções nesta execução: {motivos}. Restaram {len(dados):,} textos. Destinamos {len(tr):,} ao treino e {len(va):,} à validação, por divisão estratificada 80/20 com semente 42. Os 100 comentários externos permaneceram separados. Os números das classes foram convertidos conforme confirmação do usuário: treino 0/1/2 corresponde a teste -1/0/1. A classe 1 do treino significa neutro ou ambíguo.

Usamos somente o comentário como entrada. TF-IDF com palavras e pares de palavras, min_df=2, máximo de 100.000 atributos e frequência sublinear. Regressão Logística com C=1, class_weight=balanced, solver=lbfgs, tol=0,0001 e máximo de 1.000 iterações. Não removemos palavras de negação nem aplicamos lista de stopwords. O vocabulário foi aprendido somente no treino. Parâmetros foram fixados previamente; não houve busca de parâmetros, ajuste por erros do teste nem retreinamento após a avaliação.

## Validação interna (não confundir com teste comum)

Acurácia: {metricas['validacao_interna']['acuracia']:.2%}. F1 macro: {metricas['validacao_interna']['f1_macro']:.4f}.

## Limitações para discussão

O teste é pequeno: 40 positivos, 43 negativos e 17 neutros. O último grupo contém comentários irônicos sobre um livro e nenhuma referência positiva. Há textos mistos rotulados como neutros. Isso limita a generalização e deve ser considerado ao analisar os erros. A procedência exata do CSV e a geração dos exemplos do teste ainda precisam ser confirmadas com Rodrigo; não atribuir todos os registros a B2W/MQD sem essa confirmação. Rótulos do grupo foram preservados. A verificação de duplicatas cobre igualdade após normalização, não paráfrases. O modelo não representa o Google Cloud Natural Language. A comparação com os colegas exige que usem a mesma versão do teste e informem se incluem a descrição do produto como entrada.

## Arquivos

`metricas.json`: métricas completas, incluindo cada classe e grupo. `previsoes_teste.csv`: todas as previsões e acertos/erros. `matriz_confusao.png`: gráfico. `protocolo.json`: parâmetros, hashes e versões. `particoes.csv`: linha de origem e partição. `linhas_removidas.json`: auditoria das exclusões. `modelo_tfidf_regressao.joblib`: modelo treinado (carregar somente arquivos confiáveis).

## Fontes técnicas

- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
'''
    (out/'RESULTADOS.md').write_text(resumo,encoding='utf-8')
    print(json.dumps({'teste':metricas['teste_comum'],'validacao':metricas['validacao_interna'],'segundos_treino':treino_seg},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--treino',required=True,type=Path)
    p.add_argument('--teste',type=Path,default=ROOT/'dados/teste_ia_original.jsonl')
    p.add_argument('--saida',type=Path,default=ROOT/'resultados')
    args=p.parse_args()
    executar(args.treino,args.teste,args.saida)
