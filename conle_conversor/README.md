# Conversor Notion → Informação Técnica (CONLE)

Transpõe uma página do Notion da Consultoria Legislativa em dois documentos Word,
fiéis ao texto da página e à estética dos modelos da casa:

1. **Informação Técnica (IT)** — com timbre da Câmara, tabela-cabeçalho (TEOR DA
   SOLICITAÇÃO), abertura harmonizada, seções/subseções, tabelas, fecho e assinatura.
   Salva em `…\CONLE\STs\INFO TÉCNICA`.
2. **Minuta de proposição** — PL, PLP, PEC, PRC ou PDL, escolhida automaticamente
   pelo sentido da minuta (epígrafe, autoria, ementa, preâmbulo, articulado,
   justificação, fecho). Salva em `…\CONLE\STs\PROPOSIÇÕES`.

## Como usar (interface gráfica)

- Dê duplo-clique no atalho **“Conversor IT (CONLE)”** na Área de Trabalho
  (ou rode `abrir_conversor_conle.cmd`).
- Cole a **URL** (ou o ID) da página do Notion.
- Opcional: informe o(a) deputado(a) solicitante, o nº SISCONLE, o vocativo, o(a)
  consultor(a) e a data do fecho. Campos em branco viram placeholders editáveis.
- Marque os documentos a gerar e clique em **Gerar documentos**.

## Como usar (linha de comando)

```bash
py -c "from conle_conversor.pipeline import converter; converter('URL_DO_NOTION')"
```

## Estrutura

```
conle_conversor/
  config.py         caminhos, defaults, blocos de texto fixos (Resolução 48/93, fecho)
  notion_api.py     cliente da API do Notion (fetch recursivo de blocos)
  notion_parser.py  blocos do Notion → estrutura intermediária (rich text)
  splitter.py       separa cabeçalho, corpo da IT, minuta e justificativa
  classifier.py     detecta o tipo de proposição (PL/PLP/PEC/PRC/PDL)
  harmonizer.py     gera a abertura da IT via OpenAI (com fallback sem IA)
  richtext.py       rich text → runs do Word
  docx_builder.py   monta os .docx aplicando os estilos nomeados
  pipeline.py       orquestra tudo
  build_templates.py (re)gera os templates a partir dos modelos reais
templates/          template_IT.docx e template_proposicao.docx (timbre + estilos)
conle_conversor_gui.pyw   interface gráfica (tkinter)
```

## Requisitos

- Python do Windows (`py`), 3.11+, com `requests`, `python-docx`, `openai`
  (`py -m pip install -r conle_conversor/requirements.txt`).
- `Chave_Notion.txt` (token da integração do Notion) e `OPENAI_API_KEY` no `.env`.

## Observações

- A IA atua **apenas** na abertura (encaminhamento, transição e o campo TEOR);
  todo o corpo técnico e a minuta são transpostos fielmente do Notion.
- Os templates preservam o timbre e os estilos; para atualizá-los após mudança nos
  modelos: `py -m conle_conversor.build_templates`.
