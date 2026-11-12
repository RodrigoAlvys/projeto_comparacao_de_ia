"""Converte o teste agrupado de Rodrigo sem alterar textos ou rotulos."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROTULOS = {-1: 'negativo', 0: 'neutro', 1: 'positivo'}

def preparar(origem, destino):
    raw = origem.read_bytes()
    registros = []
    grupos = []
    for numero, linha in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not linha.strip():
            continue
        grupo = json.loads(linha)
        assert isinstance(grupo.get('descricao'), str), f'Descricao ausente na linha {numero}'
        assert isinstance(grupo.get('comentarios'), list), f'Comentarios ausentes na linha {numero}'
        grupos.append(grupo)
        for indice, par in enumerate(grupo['comentarios'], 1):
            assert isinstance(par, list) and len(par) == 2, f'Par invalido: {numero}/{indice}'
            texto, rotulo = par
            assert isinstance(texto, str) and texto.strip(), f'Texto vazio: {numero}/{indice}'
            assert type(rotulo) is int and rotulo in ROTULOS, f'Rotulo invalido: {numero}/{indice}'
            registros.append({'id': f'g{len(grupos):02d}_c{indice:02d}', 'grupo':len(grupos),
                              'texto':texto, 'rotulo':rotulo, 'classe':ROTULOS[rotulo],
                              'descricao':grupo['descricao']})
    assert registros, 'Arquivo sem comentarios'
    destino.mkdir(parents=True, exist_ok=True)
    (destino/'teste_ia_original.jsonl').write_bytes(raw)
    with (destino/'teste_padronizado.jsonl').open('w',encoding='utf-8') as arquivo:
        for registro in registros:
            arquivo.write(json.dumps(registro,ensure_ascii=False)+'\n')
    relatorio={'arquivo_origem':origem.name,'sha256_origem':hashlib.sha256(raw).hexdigest(),
               'grupos':len(grupos),'total_comentarios':len(registros),
               'classes':dict(Counter(r['classe'] for r in registros)),
               'duplicatas_exatas_ignorando_caixa_e_espacos_externos':len(registros)-len({r['texto'].strip().casefold() for r in registros}),
               'alteracao_de_rotulos':False,'treinamento_executado':False,'avaliacao_executada':False}
    (destino/'conferencia.json').write_text(json.dumps(relatorio,ensure_ascii=False,indent=2),encoding='utf-8')
    # Verifica equivalencia item a item com a fonte, inclusive ordem e contexto.
    gerados=[json.loads(x) for x in (destino/'teste_padronizado.jsonl').read_text(encoding='utf-8').splitlines()]
    esperado=[(g['descricao'],t,y) for g in grupos for t,y in g['comentarios']]
    assert [(r['descricao'],r['texto'],r['rotulo']) for r in gerados] == esperado
    print(json.dumps(relatorio,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('origem',type=Path)
    parser.add_argument('--destino',type=Path,default=Path(__file__).resolve().parent/'dados')
    args=parser.parse_args()
    preparar(args.origem,args.destino)
