# -*- coding: utf-8 -*-
"""tsje_05d: limpa residuos de tool-call que vazaram para dentro dos .md.

Alguns subagentes deixaram escapar, no fim do arquivo, o fechamento das tags
da propria ferramenta de escrita (content/invoke/function_calls, em forma
literal ou escapada com &lt;). Isso e lixo: nao faz parte da ata e
contaminaria o docx. Este script remove essas linhas do FIM do corpo.

Os padroes sao montados em tempo de execucao (evita que a tag literal no
codigo-fonte quebre a propria ferramenta que grava este arquivo).

Uso: python tsje_05d_limpar.py [--dry]
"""
import io
import os
import re
import sys

RAIZ = r'D:\TSJE_TRANSCRICOES'
LT = chr(60)   # <
GT = chr(62)   # >

# nomes de tag que nunca pertencem a uma ata
TAGS = ('content', 'invoke', 'function_calls', 'parameter', 'antml:invoke',
        'antml:parameter', 'antml:function_calls')

_alt = '|'.join(re.escape(t) for t in TAGS)
# casa a tag em forma literal (<​/content>) ou escapada (&lt;/content&gt;)
RE_LIXO = re.compile(
    r'^\s*(?:' + re.escape(LT) + r'|&lt;)\s*/?\s*(?:' + _alt + r')\s*'
    r'(?:' + re.escape(GT) + r'|&gt;)\s*$', re.I)


def limpar_texto(texto):
    linhas = texto.split('\n')
    # remove do FIM para tras: linhas de lixo e vazias entre elas
    fim = len(linhas)
    achou = 0
    while fim > 0:
        ln = linhas[fim - 1]
        if RE_LIXO.match(ln):
            achou += 1
            fim -= 1
        elif ln.strip() == '':
            # pula linhas vazias do fim (antes ou entre as tags de lixo)
            fim -= 1
        else:
            break
    if not achou:
        # so havia linhas vazias no fim: nao mexe no arquivo
        return texto, 0
    return '\n'.join(linhas[:fim]).rstrip() + '\n', achou


def main():
    dry = '--dry' in sys.argv
    tot_arq = tot_lin = 0
    for dirpath, _, files in os.walk(RAIZ):
        for nome in files:
            if not nome.endswith('.md'):
                continue
            path = os.path.join(dirpath, nome)
            with io.open(path, encoding='utf-8') as f:
                texto = f.read()
            novo, n = limpar_texto(texto)
            if not n:
                continue
            tot_arq += 1
            tot_lin += n
            print(f'{"[dry] " if dry else ""}{nome}: {n} linha(s) de lixo')
            if not dry:
                with io.open(path, 'w', encoding='utf-8') as f:
                    f.write(novo)
    print(f'{tot_arq} arquivos limpos, {tot_lin} linhas removidas')


if __name__ == '__main__':
    main()
