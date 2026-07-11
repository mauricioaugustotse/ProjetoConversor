# -*- coding: utf-8 -*-
"""Testes do pipeline determinístico do normalizador_core, com fixtures do
ruído REAL observado nas subpáginas de estatutos (ex.: PCB). Roda com pytest
ou standalone: py tests/test_normalizador.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import normalizador_core as nc  # noqa: E402


# ---------------------------------------------------------------- p02 LaTeX
def test_latex_ordinais():
    casos = {
        "Artigo $`1^{\\circ}`$ - O Partido": "Artigo 1º - O Partido",
        "Artigo $`2^{\\boldsymbol{\\circ}}`$ - O PCB": "Artigo 2º - O PCB",
        "Artigo $`\\mathbf{3}^{\\boldsymbol{\\circ}}`$ - O PCB tem": "Artigo 3º - O PCB tem",
        "art. 49 da Res.-TSE $`\\mathrm{n}^{\\circ}`$ 23.571/2018": "art. 49 da Res.-TSE nº 23.571/2018",
        "$`9{ }^{\\circ}`$": "9º",
        "por $`1 / 3`$ (um terço)": "por 1/3 (um terço)",
        "no art. 39, parágrafo $`4^{\\circ}`$ do estatuto": "no art. 39, parágrafo 4º do estatuto",
    }
    for entrada, esperado in casos.items():
        rel = {}
        saida = nc.p02_latex_para_texto(entrada, rel)
        assert saida == esperado, f"{entrada!r} -> {saida!r} (esperava {esperado!r})"
        assert "suspeitas" not in rel, f"marcou suspeita indevida em {entrada!r}"


def test_latex_nu_sem_delimitadores():
    # a API do Notion entrega equação inline como plain_text SEM os $...$
    casos = {
        "Artigo 1^{\\circ} - O Partido": "Artigo 1º - O Partido",
        "Artigo \\mathbf{3}^{\\boldsymbol{\\circ}} - O PCB tem": "Artigo 3º - O PCB tem",
        "Artigo \\mathbf{8}^{\\mathbf{o}} - É livre": "Artigo 8º - É livre",
        "9{ }^{\\circ}": "9º",
        "no art. 39, parágrafo 4^{\\circ} do presente": "no art. 39, parágrafo 4º do presente",
    }
    for entrada, esperado in casos.items():
        rel = {}
        saida = nc.p02_latex_para_texto(entrada, rel)
        assert saida == esperado, f"{entrada!r} -> {saida!r} (esperava {esperado!r})"
        assert "suspeitas" not in rel, f"suspeita indevida em {entrada!r}: {rel}"


def test_latex_ordinais_exoticos():
    casos = {
        "Art. 1^{\\underline{\\circ}} O REPUBLICANOS, pessoa jurídica":
            "Art. 1º O REPUBLICANOS, pessoa jurídica",
        "Art. 2^{\\underline{0}}. O PDT tem sede": "Art. 2º. O PDT tem sede",
        "§ 2^{ㅇ } A filiação de líderes": "§ 2º A filiação de líderes",
        "§1 ^{® } - Caberá ao Diretório": "§1º - Caberá ao Diretório",
        "RPP \\mathrm{N}^{\\circ} 0001554-73": "RPP Nº 0001554-73",
        "contribuir com R \\$ 20,00 por ano": "contribuir com R$ 20,00 por ano",
        "\\1º - Em cada um dos Diretórios": "§ 1º - Em cada um dos Diretórios",
        "texto \\end{aligned} residual": "texto  residual",
    }
    for entrada, esperado in casos.items():
        rel = {}
        saida = nc.p02_latex_para_texto(nc.p01_normalizar_chars(entrada), rel)
        assert saida == esperado, f"{entrada!r} -> {saida!r} (esperava {esperado!r})"
        assert "suspeitas" not in rel, f"suspeita indevida em {entrada!r}: {rel}"


def test_detectar_partido_democrata_vs_prd():
    md = ("O DEMOCRATA, partido político. " + "O DEMOCRATA decide. " * 15 +
          "Com a fusão do PTB com o Patriota formando o Partido Renovação "
          "Democrática (PRD), o PTB não mais existe.")
    sigla, _ = nc.detectar_partido(md)
    assert sigla == "DEMOCRATA"


def test_latex_residuo_marca_suspeita():
    rel = {}
    nc.p02_latex_para_texto("texto com resíduo \\alpha estranho", rel)
    assert rel.get("suspeitas")


def test_numero_sign():
    md = nc.p01_normalizar_chars("REGISTRO DE PARTIDO POLÍTICO № 0000081-53")
    assert "№" not in md and "nº 0000081-53" in md


def test_lixo_pje():
    rel = {}
    md = "\n".join([
        "Art. 1º - O AGIR, com sede e foro no Distrito Federal.",
        "Assinado eletronicamente por: JUAN VITOR BALDUINO NOGUEIRA - 01/12/2021 18:48:10",
        "Num. 157074473 - Pág. 4",
        "https://pje.tse.jus.br/pje/Processo/ConsultaDocumento/listView.seam?x=211201",
        "Número do documento: 21120118480945900000155772066",
        "Art. 2º - São Órgãos de Direção e Deliberação do Partido as Convenções.",
    ])
    saida = nc.p015_remover_lixo_pje(md, rel)
    assert rel["lixo_pje_linhas"] == 4
    assert saida.split("\n") == [
        "Art. 1º - O AGIR, com sede e foro no Distrito Federal.",
        "Art. 2º - São Órgãos de Direção e Deliberação do Partido as Convenções.",
    ]


# ---------------------------------------------------------------- p04 fusão
def test_fundir_artigo_quebrado():
    md = "Artigo\n9º\n- Todos os militantes são iguais, em direitos e deveres:"
    saida = nc.p04_fundir_blocos_partidos(md)
    assert saida == "Artigo 9º - Todos os militantes são iguais, em direitos e deveres:"


def test_fundir_paragrafo_orfao():
    # caso real (UP): rótulo "§ 11º" numa linha e o texto na seguinte
    md = "§ 11º\n- Deferida a filiação e registrada com a data do pedido, a Executiva enviará."
    saida = nc.p04_fundir_blocos_partidos(md)
    assert saida == ("§ 11º - Deferida a filiação e registrada com a data do pedido, "
                     "a Executiva enviará.")


def test_fundir_rotulo_com_numero():
    md = "Artigo\n12 - Compete às Células executar."
    saida = nc.p04_fundir_blocos_partidos(md)
    assert saida == "Artigo 12 - Compete às Células executar."


def test_fundir_artigo_sufixo_letra():
    # caso real (PSTU): "Artigo" / "64 B" / "- Na prestação..."
    md = "Artigo\n64 B\n- Na prestação de contas partidárias devem ser discriminadas."
    saida = nc.p04_fundir_blocos_partidos(md)
    assert saida == "Artigo 64 B - Na prestação de contas partidárias devem ser discriminadas."


def test_fundir_fragmento_percentual():
    # caso real (PSTU): "...de no mínimo" / "20%" / "(vinte por cento) do total."
    md = ("d) criação e manutenção da Fundação, sendo esta aplicação de no mínimo\n"
          "20%\n"
          "(vinte por cento) do total recebido.")
    rel = {}
    saida = nc.p045_fundir_fragmentos(md, rel)
    assert saida == ("d) criação e manutenção da Fundação, sendo esta aplicação de no mínimo "
                     "20% (vinte por cento) do total recebido.")
    assert rel["fragmentos_fundidos"] == 1


def test_fundir_ordinal_orfao():
    # caso real (PSTU): "...da Lei n." / "º" / "9096/95, os demais recursos..."
    md = ("Artigo 64 C - Descontados os 20% de que trata o inciso IV da Lei n.\n"
          "º\n"
          "9096/95, os demais recursos serão administrados pelo Diretório Nacional.")
    rel = {}
    saida = nc.p045_fundir_fragmentos(md, rel)
    assert saida == ("Artigo 64 C - Descontados os 20% de que trata o inciso IV da Lei n.º "
                     "9096/95, os demais recursos serão administrados pelo Diretório Nacional.")


def test_fragmento_nao_engole_dispositivo():
    # fragmento seguido de dispositivo novo (Maiúscula): funde só o fragmento
    md = "aplicação de no mínimo\n20%\nArtigo 65 - Novo dispositivo."
    rel = {}
    saida = nc.p045_fundir_fragmentos(md, rel)
    assert saida == "aplicação de no mínimo 20%\nArtigo 65 - Novo dispositivo."
    # número de artigo em linha própria (rótulo, não fragmento) fica intacto p/ o p04
    md2 = "Artigo\n64 B\n- texto."
    rel2 = {}
    assert nc.p045_fundir_fragmentos(md2, rel2) == md2


# ---------------------------------------------------------------- p05 split
def test_separar_artigos_colados():
    md = ("Artigo 42 - O Secretariado é constituído de 3 militantes. "
          "Artigo 43 - O Secretariado dirige as atividades.")
    saida = nc.p05_separar_artigos_colados(md)
    assert saida.count("\n") == 1
    l1, l2 = saida.split("\n")
    assert l1.startswith("Artigo 42") and l2.startswith("Artigo 43")


def test_separar_paragrafo_colado():
    md = ("Artigo 46 - O número de membros é fixado pela Conferência. "
          "Parágrafo Primeiro - Na eleição observar-se-ão os critérios.")
    saida = nc.p05_separar_artigos_colados(md)
    assert "\nParágrafo Primeiro" in saida


def test_nao_separa_referencia_interna():
    md = "conforme previsto no Artigo 39, parágrafo 4º do presente estatuto."
    assert nc.p05_separar_artigos_colados(md) == md
    md2 = "aplicando-se o disposto no art. 15 desta norma ao caso."
    assert nc.p05_separar_artigos_colados(md2) == md2


def test_artigo_traco_colado():
    assert nc.p05_separar_artigos_colados("Artigo 12-O militante que infringir") == \
        "Artigo 12 - O militante que infringir"
    assert nc.p05_separar_artigos_colados("Artigo 65-Na Casa Legislativa, o integrante") == \
        "Artigo 65 - Na Casa Legislativa, o integrante"
    # sufixo real de dispositivo não é tocado
    assert nc.p05_separar_artigos_colados("Artigo 37-A - Fica assegurada a paridade") == \
        "Artigo 37-A - Fica assegurada a paridade"


def test_percentual_com_backslash():
    rel = {}
    saida = nc.p02_latex_para_texto("repassará 50 \\% (cinquenta por cento) e 3 \\\\% ao mês", rel)
    assert saida == "repassará 50% (cinquenta por cento) e 3% ao mês"
    assert "suspeitas" not in rel


def test_aspas_e_fracao():
    md = nc.p01_normalizar_chars('nas alíneas " d "," e " e " g " com 2 / 3 dos membros')
    assert '"d"' in md and '"e"' in md and '"g"' in md and "2/3" in md


def test_aspas_idempotente_nao_colapsa_conector():
    # texto JÁ limpo: '"f" e "g"' deve permanecer intacto em qualquer repassada
    limpo = 'previstas nas alíneas "d", "e", "f" e "g" deverá ser registrada'
    assert nc.p01_normalizar_chars(limpo) == limpo


# ---------------------------------------------------------------- p06 headings
def test_fundir_heading_capitulo():
    md = "## Capítulo I\n## Do Partido, seus Objetivos e Símbolos:"
    saida = nc.p06_fundir_headings_duplicados(md)
    assert saida == "## Capítulo I — Do Partido, seus Objetivos e Símbolos"


def test_fundir_heading_com_paragrafo_nome():
    md = "## Capítulo II\nDa inscrição e da Militância Partidária;"
    saida = nc.p06_fundir_headings_duplicados(md)
    assert saida == "## Capítulo II — Da inscrição e da Militância Partidária"


def test_nao_funde_heading_com_artigo():
    md = "## Capítulo IX\nArtigo 45 - Compete aos Comitês."
    saida = nc.p06_fundir_headings_duplicados(md)
    assert "— Artigo" not in saida


# ---------------------------------------------------------------- p08 recorte
_CABECALHO_PCB = """## REGISTRO DE PARTIDO POLÍTICO nº 0000081-53.1994.6.00.0000
Relator: Ministro Mauro Campbell Marques
Requerente: Partido Comunista Brasileiro (PCB) - Nacional
Advogado: Messias Silva Manarim - OAB: 47779/SC
Em sessão realizada em 16 de agosto de 2022, este Tribunal, por unanimidade, deferiu o pedido de anotação das alterações estatutárias do Partido Comunista Brasileiro (PCB), conforme ementa a seguir transcrita:
PARTIDO POLÍTICO. PEDIDO DE REGISTRO DAS ALTERAÇÕES ESTATUTÁRIAS. ART. 49 DA RES.-TSE nº 23.571/2018. DEFERIMENTO.
# Partido Comunista Brasileiro (PCB)
## ESTATUTO
(registrado em Cartório e homologado pelo TSE)
## Capítulo I
## Do Partido, seus Objetivos e Símbolos:
Artigo 1º - O Partido Comunista Brasileiro (PCB), pessoa jurídica de direito privado.
Artigo 76 - Este Estatuto entra em vigor na data de sua aprovação.
São Paulo, 25 de fevereiro de 2022.
EDMILSON SILVA - SECRETÁRIO GERAL
[IMAGEM n=1 kind=file]
MESSIAS SILVA MANARIM ADVOGADO - OAB/SC nº 47.779
[IMAGEM n=2 kind=file] """


def test_recorte_inicio_e_fim():
    rel = {}
    md, ini, fim = nc.p08_recortar_escopo(_CABECALHO_PCB, rel)
    assert md.startswith("# Partido Comunista Brasileiro (PCB)")
    assert "REGISTRO DE PARTIDO" not in md
    assert any("Relator" in l for l in ini)
    assert rel.get("data_aprovacao") == "16.8.2022"
    assert rel.get("rpp") == "0000081-53.1994.6.00.0000"
    # fim: local/data, assinaturas e imagens saem; o último artigo fica
    assert "entra em vigor" in md
    assert "São Paulo, 25 de fevereiro" not in md
    assert "MESSIAS" not in md
    assert any("EDMILSON" in l for l in fim)


def test_recorte_sem_ancora_nao_corta():
    rel = {}
    md_original = "Texto qualquer sem estatuto.\nOutra linha."
    md, ini, fim = nc.p08_recortar_escopo(md_original, rel)
    assert md == md_original and not ini
    assert "não identificado" in rel.get("recorte_inicio", "")


def test_recorte_prefixo_sem_marcadores_nao_corta():
    rel = {}
    md_original = "Introdução inocente do documento.\n# Estatuto do Partido X\nArtigo 1º - texto."
    md, ini, _ = nc.p08_recortar_escopo(md_original, rel)
    assert "Introdução inocente" in md and not ini


def test_recorte_nao_come_dispositivos():
    # caso PT: âncora só é encontrada tarde (Título I em grego virou irreconhecível
    # em versões antigas); se o prefixo tem artigos em sequência, NADA é cortado
    # ordinais degradados ("Art. 1 e.", "Art. 29.") impedem a âncora por Art. 1º;
    # a única âncora acha o Capítulo TARDE — o gate deve suspender o corte
    md = "\n".join([
        "## TRIBUNAL SUPERIOR ELEITORAL",
        "Relator: Ministro X",
        "Requerente: Partido Y",
        "Em sessão realizada em 1 de junho de 2023, deferiu a anotação das alterações estatutárias.",
        "Art. 1 e. O Partido é uma associação voluntária.",
        "Art. 29. O Partido tem duração indeterminada.",
        "Art. 14. São deveres do filiado participar.",
        "## CAPÍTULO I DISPOSIÇÕES GERAIS",
        "Art. 15. A unidade do Partido.",
    ])
    rel = {}
    saida, ini, _ = nc.p08_recortar_escopo(md, rel)
    assert not ini and "Art. 1 e. O Partido é uma associação voluntária." in saida
    assert "SUSPENSO" in rel.get("recorte_inicio", "")


def test_ancora_titulo_grego_e_colado():
    # "Τίτυιο ι" (grego) e "CAPÍTULOI" (colado) devem ancorar após p01
    md = "\n".join([
        "## TRIBUNAL SUPERIOR ELEITORAL",
        "Relator: Ministro X",
        "Em sessão realizada em 1 de junho de 2023, anotação das alterações estatutárias.",
        "## Τίτυιο ι  DO PARTIDO, SEDE, OBJETIVO",
        "## CAPÍTULOI  DA DURAÇÃO, SEDE E FORO",
        "Art. 1 e. O Partido dos Trabalhadores (PT) é uma associação.",
    ])
    rel = {}
    saida, ini, _ = nc.p08_recortar_escopo(nc.p01_normalizar_chars(md), rel)
    assert any("Relator" in l for l in ini)
    assert saida.split("\n")[0].startswith("## T")  # Título I mantido


def test_ancora_estende_para_tras():
    # caso UP: "# UNIDADE POPULAR ESTATUTO" logo antes do "## Capítulo I"
    md = "\n".join([
        "## CERTIDÃO DE JULGAMENTO",
        "REGISTRO DE PARTIDO POLÍTICO nº 0600412-09.2019.6.00.0000",
        "REQUERENTE: UNIDADE POPULAR (UP) - NACIONAL",
        "Em sessão realizada em 10.12.2019, DEFERIMENTO do registro.",
        "# UNIDADE POPULAR ESTATUTO",
        "## Capítulo I",
        "## Do Partido, sede e princípios básicos",
        "Art. 1º - A Unidade Popular é um partido.",
    ])
    rel = {}
    saida, ini, _ = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[0] == "# UNIDADE POPULAR ESTATUTO"
    assert any("CERTIDÃO" in l for l in ini)


def test_ancora_preambulo():
    md = "\n".join([
        "## REGISTRO DE PARTIDO POLÍTICO nº 0001658-95.1996.6.00.0000",
        "Relator: Ministro F",
        "Em sessão realizada em 23 de maio de 2024, deferiu a anotação estatutária.",
        "## PREÂMBULO",
        "O PODEMOS se apresenta como uma resposta aos anseios dos cidadãos.",
        "## TÍTULO I - DO PARTIDO E DOS OBJETIVOS",
        "Art. 1º - O PODEMOS, pessoa jurídica de direito privado.",
    ])
    rel = {}
    saida, ini, _ = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[0] == "## PREÂMBULO"
    assert any("Relator" in l for l in ini)


def test_nao_funde_estatuto_com_rotulo_estrutural():
    md = "\n".join([
        "# ESTATUTO",
        "## TÍTULO I DA DENOMINAÇÃO",
        "Art. 1. O União Brasil, pessoa jurídica.",
    ])
    rel = {}
    saida = nc.p10_hierarquia(md, rel)
    assert "Estatuto — TÍTULO" not in saida
    assert "# ESTATUTO" in saida.split("\n")[0]


def test_lixo_carimbo_protocolo():
    rel = {}
    md = "\n".join([
        "## Capítulo I",
        "Nº de Protocolo",
        "135960",
        "Registro de Pessoas Juridicas",
        "## Do Partido, sede e princípios básicos",
    ])
    saida = nc.p015_remover_lixo_pje(md, rel)
    assert rel["lixo_pje_linhas"] == 3
    assert "Protocolo" not in saida and "135960" not in saida


def test_recorte_fim_corta_fecho_inteiro():
    # zona de assinaturas com nomes/cargos que a heurística linha-a-linha não pegava
    md = "\n".join([
        "# Estatuto do SOLIDARIEDADE",
        "Art. 131 - Este Estatuto entra em vigor na data de seu registro.",
        "Brasília, 28 de abril de 2022.",
        "LUIZ ANTONIO ADRIANO DA SILVA",
        "SECRETÁRIO-GERAL NACIONAL DO SOLIDARIEDADE",
        "ADVOGADA:",
        "Marilda de Paula Silveira",
        "OAB/DF - 49.068",
    ])
    rel = {}
    saida, _, fim = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[-1].startswith("Art. 131")
    assert len(fim) == 6


def test_recorte_fim_preserva_alineas_do_ultimo_artigo():
    md = "\n".join([
        "# Estatuto X",
        "Art. 100 - São fontes de receita do Partido:",
        "a) contribuições dos filiados;",
        "b) doações permitidas em lei.",
        "Parágrafo único - Os recursos serão contabilizados.",
        "João da Silva",
        "Presidente Nacional",
    ])
    rel = {}
    saida, _, fim = nc.p08_recortar_escopo(md, rel)
    assert "b) doações permitidas em lei." in saida
    assert "Parágrafo único - Os recursos serão contabilizados." in saida
    assert fim == ["João da Silva", "Presidente Nacional"]


def test_recorte_fim_tabela_de_assinatura_digital():
    # caso MISSÃO: a caixinha de assinatura digital do PDF virou tabela no fecho
    md = "\n".join([
        "# Estatuto da MISSÃO",
        "Art. 91 - Este Estatuto entra em vigor na data de seu registro.",
        "## RENAN ANTONIO FERREIRA DOS SANTOS PRESIDENTE DA EXECUTIVA NACIONAL",
        "[TABELA]",
        "  | VICTOR SOUZA | Digitally signed by VICTOR |",
        "  | COUTO:4502577685 | Date: 2026.03.04 14:56:13 |",
        "## VICTOR SOUZA LOPES DE COUTO",
    ])
    rel = {}
    saida, _, fim = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[-1].startswith("Art. 91")
    assert len(fim) == 5


def test_recorte_fim_tabela_de_conteudo_bloqueia():
    # anexo legítimo com tabela SEM marcador de fecho: mantém (conservador)
    md = "\n".join([
        "# Estatuto Y",
        "Art. 50 - Este Estatuto entra em vigor na data do registro.",
        "Anexo I - Tabela de contribuições por faixa",
        "[TABELA]",
        "  | Faixa | Valor |",
        "  | A | 10 |",
    ])
    rel = {}
    saida, _, fim = nc.p08_recortar_escopo(md, rel)
    assert "Anexo I" in saida and "| A | 10 |" in saida


# ---------------------------------------------------------------- p09/p10
def test_remover_imagens():
    rel = {}
    md = nc.p09_remover_imagens("texto\n[IMAGEM n=1 kind=file] legenda\nfim", rel)
    assert "[IMAGEM" not in md and rel["imagens_removidas"] == 1


def test_manter_imagens_vira_placeholder():
    rel = {}
    md = nc.p09_remover_imagens("[IMAGEM n=3 kind=file] selo", rel, manter=True)
    assert md == "> 🖼️ Figura 3 — selo"


def test_hierarquia_preambulo_e_niveis():
    md = "\n".join([
        "# Partido Comunista Brasileiro (PCB)",
        "## ESTATUTO",
        "(registrado em Cartório e homologado pelo TSE)",
        "## Capítulo I — Do Partido",
        "Artigo 1º - texto.",
        "### Seção II — Das Finanças",
        "## Artigo 50 - falso heading",
    ])
    rel = {}
    saida = nc.p10_hierarquia(md, rel)
    linhas = saida.split("\n")
    assert linhas[0] == "# Estatuto — Partido Comunista Brasileiro (PCB)"
    assert "## Capítulo I — Do Partido" in linhas
    assert "### Seção II — Das Finanças" in linhas
    assert "Artigo 50 - falso heading" in linhas  # rebaixado a parágrafo
    assert "## Artigo 50" not in saida


# ---------------------------------------------------------------- documentos genéricos
def test_detectar_tipo_documento():
    normativo = "\n".join(f"Art. {i}º - Disposição número {i}." for i in range(1, 10))
    assert nc.detectar_tipo_documento(normativo) == "normativo"
    generico = "Este relatório apresenta os resultados da avaliação anual.\nSeção 2."
    assert nc.detectar_tipo_documento(generico) == "generico"


def test_prompt_por_tipo():
    s_norm = nc._SYSTEM_REVISAO.replace("{contexto}", nc._CONTEXTO_DOC["normativo"])
    s_gen = nc._SYSTEM_REVISAO.replace("{contexto}", nc._CONTEXTO_DOC["generico"])
    assert "NORMATIVOS" in s_norm and "relatórios" in s_gen
    assert "{contexto}" not in s_norm and "estatutos de partidos" not in s_gen


def test_ancora_documento_generico():
    md = "\n".join([
        "## OFÍCIO DE ENCAMINHAMENTO Nº 42",
        "Em sessão realizada em 5 de maio de 2026, o órgão deliberou pelo DEFERIMENTO.",
        "Encaminho a Vossa Senhoria o relatório anexo para anotação das alterações estatutárias.",
        "# RELATÓRIO ANUAL DE ATIVIDADES 2026",
        "Este documento apresenta as atividades desenvolvidas no exercício.",
    ])
    rel = {}
    saida, ini, _ = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[0] == "# RELATÓRIO ANUAL DE ATIVIDADES 2026"
    assert any("OFÍCIO" in l for l in ini)


def test_recorte_fim_documento_nao_articulado():
    md = "\n".join([
        "# PARECER TÉCNICO",
        "A análise conclui pela viabilidade da proposta apresentada.",
        "Brasília, 5 de maio de 2026.",
        "JOÃO DA SILVA",
        "OAB/DF 12.345",
    ])
    rel = {}
    saida, _, fim = nc.p08_recortar_escopo(md, rel)
    assert saida.split("\n")[-1] == "A análise conclui pela viabilidade da proposta apresentada."
    assert len(fim) == 3


# ---------------------------------------------------------------- pipeline / gate
def test_pipeline_completo_pcb():
    limpo, rel, ini, fim = nc.limpar_md(_CABECALHO_PCB)
    assert rel["artigos_pos_recorte"] == rel["artigos_final"]
    assert rel["ruido_final"]["latex"] == 0
    assert rel["ruido_final"]["imagens"] == 0
    assert limpo.splitlines()[0].startswith("# Estatuto — Partido Comunista Brasileiro")


def test_contar_artigos():
    md = "Artigo 1º - a. Artigo 2º - b.\nno art. 5º da lei\nART. 49 DA RES"
    assert nc.contar_artigos(md) == 4


def test_gate_fidelidade():
    assert nc.gate_fidelidade("Artigo 9º - Todos os militantes são iguais",
                              "Artigo 9º - Todos os militantes são iguais")
    # correção de OCR legítima (2O22 -> 2022 muda dígitos: gate deve REJEITAR por segurança)
    assert not nc.gate_fidelidade("no ano de 2O22 tudo mudou", "no ano de 2022 tudo mudou")
    # remoção de conteúdo -> rejeita
    assert not nc.gate_fidelidade(
        "Artigo 10 - O militante tem direito a voto e a ser votado nas instâncias",
        "Artigo 10 - O militante tem direito a voto")
    # número alterado -> rejeita
    assert not nc.gate_fidelidade("prazo de 30 dias", "prazo de 60 dias")


def test_rodapes_repetidos():
    rel = {}
    rodape = "Sede Operacional: Av. Padre Pereira de Andrade, 758 - Tel/Fax: (11) 3023-2727"
    md = "\n".join([
        "Art. 1º - A Democracia Cristã é um partido político.",
        rodape,
        "Art. 2º - O partido tem sede em São Paulo.",
        "> " + rodape,
        "Art. 3º - São órgãos do partido as convenções.",
    ])
    saida = nc.p016_rodapes_repetidos(md, rel)
    assert rel["rodapes_removidos"] == 2
    assert "Tel/Fax" not in saida and saida.count("Art.") == 3
    # rodapé que aparece SÓ UMA vez não é removido (pode ser conteúdo)
    rel2 = {}
    md2 = "Art. 1º - texto.\n" + rodape + "\nArt. 2º - texto."
    assert nc.p016_rodapes_repetidos(md2, rel2) == md2


def test_numeros_de_pagina_e_reunificacao():
    rel = {}
    md = "\n".join([
        "Art. 10 - A Convenção Nacional é o órgão máximo, cabendo a ela deliberar sobre",
        "3",
        "as matérias de organização interna e o programa do partido.",
        "Art. 11 - Compete ao Diretório Nacional.",
        "00185497",
    ])
    saida = nc.p046_numeros_de_pagina(nc.p015_remover_lixo_pje(md, rel), rel)
    linhas = [l for l in saida.split("\n") if l.strip()]
    assert linhas[0].endswith("deliberar sobre as matérias de organização interna e o programa do partido.")
    assert "00185497" not in saida and rel["paragrafos_reunificados"] == 1


def test_ordinais_dispositivos():
    assert nc.p055_ordinais_dispositivos("Art. 7 0º - Compete ao Diretório.") == \
        "Art. 70 - Compete ao Diretório."
    assert nc.p055_ordinais_dispositivos("Artigo 13º - Das convenções.") == \
        "Artigo 13 - Das convenções."
    assert nc.p055_ordinais_dispositivos("Art. 5 - Da filiação.") == "Art. 5º - Da filiação."
    assert nc.p055_ordinais_dispositivos("§ 3 - O prazo será de dez dias.") == \
        "§ 3º - O prazo será de dez dias."
    assert nc.p055_ordinais_dispositivos("na forma do Artigo 13º deste Estatuto") == \
        "na forma do Artigo 13 deste Estatuto"
    assert nc.p055_ordinais_dispositivos("I- primeiro inciso") == "I - primeiro inciso"
    assert nc.p055_ordinais_dispositivos("a)primeira alínea") == "a) primeira alínea"
    # 1º-9º corretos e referências sem separador ficam como estão
    assert nc.p055_ordinais_dispositivos("Art. 5º - ok") == "Art. 5º - ok"


def test_pontuacao_fullwidth_e_percentual():
    md = nc.p01_normalizar_chars("1/3（um terço）dos membros，com 5 0% dos votos．Fim")
    assert "1/3 (um terço) dos membros, com 50% dos votos. Fim" in md.replace("  ", " ")


def test_gate_delecao_micro_correcao():
    # grafia impossível: 1 replace de 1 char -> aceita
    assert nc.gate_delecao("O Paptido tem sede própria", "O Partido tem sede própria")
    # mas troca de dígito continua proibida
    assert not nc.gate_delecao("prazo de 30 dias", "prazo de 80 dias")


# ---------------------------------------------------------------- gate revisão final
def test_gate_delecao_remove_ruido_incrustado():
    # caso real (UP): carimbo de cartório no meio de um inciso
    antes = ("I - representar o Partido nas atividades políticas e perante a Just "
             "1º C , Ofício de Brasilia-DF ąche PRotocolo passivamente, judicial e "
             "extrajudicialmente;")
    depois = ("I - representar o Partido nas atividades políticas e perante a "
              "Justiça, passivamente, judicial e extrajudicialmente;")
    # 'Justiça' completa a palavra cortada 'Just' -> tem inserção ('ica') -> REJEITA
    assert not nc.gate_delecao(antes, depois)
    # remoção pura do carimbo (sem completar palavra) -> ACEITA
    depois_puro = ("I - representar o Partido nas atividades políticas e perante a Just "
                   "passivamente, judicial e extrajudicialmente;")
    assert nc.gate_delecao(antes, depois_puro)


def test_gate_delecao_respacamento():
    assert nc.gate_delecao("A Secre taria Nacional adminis tra os recursos.",
                           "A Secretaria Nacional administra os recursos.")
    assert nc.gate_delecao("prazo de 30 (trinta) dias , contados", "prazo de 30 (trinta) dias, contados")


def test_gate_delecao_rejeita_conteudo_novo():
    assert not nc.gate_delecao("prazo de 30 dias", "prazo de 60 dias")
    assert not nc.gate_delecao("O partido tem sede", "O partido tem sede em Brasília")
    assert not nc.gate_delecao("Art. 5º - Compete ao Diretório", "Art. 5º-A - Compete ao Diretório")


def test_gate_delecao_limites():
    # remoção acima de 30% -> rejeita
    assert not nc.gate_delecao("um dois tres quatro cinco seis sete oito nove dez",
                               "um dois tres quatro")
    # linha inteira só some se curta
    assert nc.gate_delecao("165725", "")
    assert not nc.gate_delecao("Artigo 10 - " + "conteúdo normativo relevante " * 5, "")


# ---------------------------------------------------------------- partido/título
def test_detectar_partido_pcb():
    sigla, conf = nc.detectar_partido(
        "# Partido Comunista Brasileiro (PCB)\n## ESTATUTO\nArtigo 1º - O PCB ...\n"
        "O PCB educa seus militantes. O PCB tem por objetivo.")
    assert sigla == "PCB" and conf >= 2


def test_detectar_partido_pcdob_nao_confunde():
    sigla, _ = nc.detectar_partido(
        "Estatuto do Partido Comunista do Brasil (PCdoB).\n"
        "O PCdoB é regido... O PCdoB tem... O PCdoB mantém...")
    assert sigla == "PCdoB"


def test_detectar_partido_ambiguo():
    sigla, conf = nc.detectar_partido("Documento sem partido nenhum citado.")
    assert sigla is None and conf == 0


def test_montar_titulo():
    assert nc.montar_titulo("PCB", "aprovado em", "16.8.2022") == \
        "Estatuto do PCB — aprovado em 16.8.2022"
    assert nc.montar_titulo("REDE", "", "") == "Estatuto da REDE"


def test_detectar_data_do_titulo():
    rot, data = nc.detectar_data("001_Estatuto_do_Partido_de_3_3_2022_PDF", {})
    assert (rot, data) == ("de", "3.3.2022")
    rot, data = nc.detectar_data("001_x_aprovado_em_15_8_2024", {})
    assert (rot, data) == ("aprovado em", "15.8.2024")
    rot, data = nc.detectar_data("001_x", {"data_aprovacao": "16.8.2022"})
    assert (rot, data) == ("aprovado em", "16.8.2022")


# ---------------------------------------------------------------- runner standalone
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ok = falhas = 0
    for nome, fn in sorted(globals().items()):
        if nome.startswith("test_") and callable(fn):
            try:
                fn()
                ok += 1
                print(f"PASS {nome}")
            except AssertionError as exc:
                falhas += 1
                print(f"FAIL {nome}: {exc}")
    print(f"\n{ok} ok, {falhas} falha(s)")
    sys.exit(1 if falhas else 0)
