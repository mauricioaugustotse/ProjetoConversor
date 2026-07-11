# -*- coding: utf-8 -*-
"""od_14 — Organização e renomeação da pasta OneDrive\\Documentos\\03 - Financeiro\\IRPF.

Regras da base (memória onedrive-documentos-organizacao):
- acentos preservados (NFC), convenção "Tipo - descrição - pessoa/órgão - data";
- pastas por ANO-BASE; subpasta por pessoa quando houver docs de terceiros;
- sem dedup (nenhuma exclusão); log de reversão CSV; \\?\ para caminhos.

Identificações feitas por leitura de conteúdo (headers .DEC/.REC, PDFs, imagens):
- CPF 769.510.067-68 = Mario Augusto Vieira; 111.618.258-00 = Luiz Celso Vieira;
- arquivos rotulados "Maurício" em 2013 que na verdade são do Mario;
- exportações do Carnê-Leão rotuladas como "Declaração IRPF";
- versões original/retificadora extraídas dos recibos/headers.

Uso: python od_14_irpf_organiza.py [--apply]   (sem --apply = dry-run)
"""
import csv
import os
import sys
import unicodedata
from datetime import datetime

BASE = r"C:\Users\mauri\OneDrive\Documentos\03 - Financeiro\IRPF"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "od_14_irpf_log.csv")

MKDIRS = [
    "2013\\Mario Augusto Vieira",
    "2013\\Luiz Celso Vieira",
    "2023\\Ana Carolina",
    "2025\\Gabriel",
]

DIR_RENAMES = [
    ("2024\\Carol", "2024\\Ana Carolina"),
    ("2025\\Carol", "2025\\Ana Carolina"),
]

# (origem_relativa, destino_relativo) — ordem importa nas cadeias de liberação de nome
OPS = [
    # ---- 2011 ----
    ("2011\\Artigo - planos de previdência privada - IRPF.doc",
     "Documentos antigos\\Artigo - planos de previdência privada e IRPF.doc"),
    ("2011\\Declaração IRPF 2012 ano-base 2011 - Maurício Augusto Chiaramonte Vieira.pdf",
     "2011\\Declaração IRPF 2012 ano-base 2011 - via MIDAS - Maurício.pdf"),
    ("2011\\Declaração IRPF 2012 ano-base 2011 - Maurício Augusto Chiaramonte Vieira (2).pdf",
     "2011\\Declaração IRPF 2012 ano-base 2011 - Maurício.pdf"),
    # ---- 2012 ----
    ("2012\\02533258113-IRPF-A-2013-2012-ORIGI.DEC",
     "2012\\Declaração IRPF 2013 ano-base 2012 - Maurício.dec"),
    ("2012\\02533258113-IRPF-A-2013-2012-ORIGI.REC",
     "2012\\Recibo - entrega IRPF 2013 - Maurício.rec"),
    ("2012\\Declaração IRPF 2013 ano-base 2012 - Maurício Augusto Chiaramonte Vieira (2).pdf",
     "2012\\Declaração IRPF 2013 ano-base 2012 - via MIDAS - Maurício.pdf"),
    ("2012\\Declaração IRPF 2013 ano-base 2012 - Maurício Augusto Chiaramonte Vieira.pdf",
     "2012\\Declaração IRPF 2013 ano-base 2012 - Maurício.pdf"),
    ("2012\\Recibo - Mauricio - entrega IRPF 2013 - Maurício Augusto Chiaramonte Vieira.pdf",
     "2012\\Recibo - entrega IRPF 2013 - Maurício.pdf"),
    # ---- 2013 ----
    ("2013\\02533258113-LEAO-2014-2014-COPIA-SEG.BAK",
     "2014\\Leão 2014 - cópia de segurança - Maurício.bak"),
    ("2013\\02533258113-LEAO-2014-2014-COPIA-SEG.DBK",
     "2014\\Leão 2014 - cópia de segurança - Maurício.dbk"),
    ("2013\\76951006768-IRPF-A-2014-2013-ORIGI.REC",
     "2013\\Mario Augusto Vieira\\Recibo - entrega IRPF 2014 original - Mario Augusto Vieira.rec"),
    ("2013\\76951006768-IRPF-A-2014-2013-RETIF.BAK",
     "2013\\Mario Augusto Vieira\\Declaração IRPF 2014 ano-base 2013 - retificadora - Mario Augusto Vieira.bak"),
    ("2013\\76951006768-IRPF-A-2014-2013-RETIF.RAK",
     "2013\\Mario Augusto Vieira\\Recibo - entrega IRPF 2014 retificadora - Mario Augusto Vieira.rak"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 (2).dec",
     "2013\\Luiz Celso Vieira\\Declaração IRPF 2014 ano-base 2013 - Luiz Celso Vieira.dec"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 (3).dec",
     "2013\\Mario Augusto Vieira\\Declaração IRPF 2014 ano-base 2013 - original - Mario Augusto Vieira.dec"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 - Maurício (2).dec",
     "2013\\Mario Augusto Vieira\\Declaração IRPF 2014 ano-base 2013 - retificadora 1 - Mario Augusto Vieira.dec"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 - Maurício.dec",
     "2013\\Mario Augusto Vieira\\Declaração IRPF 2014 ano-base 2013 - retificadora 2 - Mario Augusto Vieira.dec"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 - Maurício.pdf",
     "2013\\Declaração IRPF 2014 ano-base 2013 - via MIDAS - Maurício.pdf"),
    ("2013\\Declaração IRPF 2014 ano-base 2013 - Maurício (2).pdf",
     "2013\\Declaração IRPF 2014 ano-base 2013 - Maurício.pdf"),
    ("2013\\Declaração IRPF 2014 ano-base 2013.dbk",
     "2013\\Declaração IRPF 2014 ano-base 2013 - Maurício.dbk"),
    ("2013\\Declaração IRPF 2014 ano-base 2013.dec",
     "2013\\Declaração IRPF 2014 ano-base 2013 - Maurício.dec"),
    ("2013\\Declaração IRPF 2015 ano-base 2014.dec",
     "2014\\Leão 2014 - exportação - Maurício.dec"),
    ("2013\\Recibo - entrega IRPF 2014 (2).rec",
     "2013\\Luiz Celso Vieira\\Recibo - entrega IRPF 2014 - Luiz Celso Vieira.rec"),
    ("2013\\Recibo - entrega IRPF 2014.rec",
     "2013\\Recibo - entrega IRPF 2014 - Maurício.rec"),
    ("2013\\Recibo - Mauricio - entrega IRPF 2014 - Maurício.pdf",
     "2013\\Recibo - entrega IRPF 2014 - Maurício.pdf"),
    # ---- 2014 ----
    ("2014\\02533258113-LEAO-2015-2015-COPIA-SEG.BAK",
     "2015\\Leão 2015 - cópia de segurança - Maurício.bak"),
    ("2014\\02533258113-LEAO-2015-2015-COPIA-SEG.DBK",
     "2015\\Leão 2015 - cópia de segurança - Maurício.dbk"),
    ("2014\\Certidão negativa de débitos - Maurício Augusto Chiaramonte Vieira - 05-04-2016.pdf",
     "2015\\Certidão negativa de débitos - Receita Federal - Maurício - 05-04-2016.pdf"),
    ("2014\\Certidão negativa de débitos - TABELIONATO DE PROTESTO DE TITULO - 05-04-2016.pdf",
     "2015\\Certidão negativa de débitos - Tabelionato de Protesto de Títulos - 05-04-2016.pdf"),
    ("2014\\Comprovante - rendimentos anuais - Bradesco.pdf",
     "2015\\Informe de rendimentos - Bradesco - ano-base 2015 - Tabelionato de Protesto de Títulos.pdf"),
    ("2014\\Informe de rendimentos - CAIXA DE ASSISTÊNCIA DOS ADVOGADOS DE MINAS GERAIS - ano-base 2015 - Maurício.pdf",
     "2015\\Informe de rendimentos - CAA-MG - ano-base 2015 - Maurício.pdf"),
    ("2014\\Informe de rendimentos - EMPRESA BRASILEIRA DE CORREIOS E TELEGRAFOS - ano-base 2015 - Maurício.pdf",
     "2015\\Informe de rendimentos - Correios ECT - ano-base 2015 - Maurício.pdf"),
    ("2014\\Pedido PRL - MAURÍCIO AUGUSTO CHIARAMONTE VIEIRA.pdf",
     "2015\\Termo de adesão - PLR ECT exercícios 2013-2015 - Correios - Maurício - 10-02-2015.pdf"),
    ("2014\\PLR_ECT_Declaração de rendimentos.pdf",
     "2015\\Informe de rendimentos - PLR - Correios ECT - Maurício.pdf"),
    ("2014\\POSTALIS_ Previdência complementar_Despesas efetivadas em 2015.pdf",
     "2015\\Declaração para IR - contribuições autopatrocínio Postalprev - Postalis - ano-base 2015 - Maurício.pdf"),
    ("2014\\Relatório - situação fiscal CNPJ 09.339.255 0001-58 - Receita Federal - 05-04-2016.pdf",
     "2015\\Relatório - situação fiscal - Tabelionato de Protesto de Títulos - 05-04-2016.pdf"),
    ("2014\\Relatório complementar - situação fiscal CNPJ 09.339.255 0001-58 - Receita Federal - 05-04-2016.pdf",
     "2015\\Relatório complementar - situação fiscal - Tabelionato de Protesto de Títulos - 05-04-2016.pdf"),
    ("2014\\Relatório de situação fiscal CPF 025.332.581-13 - Maurício.pdf",
     "2015\\Relatório - situação fiscal - Maurício - 05-04-2016.pdf"),
    ("2014\\Tabela FIPE - Polo Sedan 2013 - 12-2015 - R$ 35.554,00.pdf",
     "2015\\Tabela Fipe - VW Polo Sedan 2013 - 12-2015.pdf"),
    ("2014\\Declaração IRPF 2015 ano-base 2014.dec",
     "2014\\Declaração IRPF 2015 ano-base 2014 - Maurício.dec"),
    # ---- 2015 ----
    ("2015\\Declaração IRPF 2015 ano-base 2014 - Maurício.pdf",
     "2014\\Declaração IRPF 2015 ano-base 2014 - Maurício.pdf"),
    ("2015\\Declaração IRPF 2016 ano-base 2015 - Maurício (2).dec",
     "2015\\Leão 2015 - exportação - Maurício (2).dec"),
    ("2015\\Declaração IRPF 2016 ano-base 2015 - Maurício.dec",
     "2015\\Leão 2015 - exportação - Maurício.dec"),
    ("2015\\Leão - cópia de segurança - IRPF 2016 ano-base 2015 - Maurício.dbk",
     "2016\\Leão 2016 - cópia de segurança - Maurício (2).dbk"),
    ("2015\\Informe de rendimentos - Sistema Dirf - ano-base 2015.pdf",
     "2015\\Informe de rendimentos - Sistema Dirf - ano-base 2015 - Maurício.pdf"),
    # ---- 2016 ----
    ("2016\\7.pdf",
     "2016\\Declaração para IR - contribuições autopatrocínio Postalprev - Postalis - ano-base 2016 - Maurício.pdf"),
    ("2016\\TERMO DE PORTABILIDADE.pdf",
     "2016\\Termo de portabilidade - Postalis para Funpresp-Jud - Maurício - R$ 62.013,42 - 15-07-2016.pdf"),
    ("2016\\DARF - IRPF cód. 0190 - 29-01-2016.pdf",
     "2016\\Comprovante BB - pagamento DARF Carnê-Leão - 29-01-2016.pdf"),
    ("2016\\DARF - IRPF cód. 0190 - 29-02-2016.pdf",
     "2016\\Comprovante BB - pagamento DARF Carnê-Leão - 29-02-2016.pdf"),
    ("2016\\DARF - IRPF cód. 0190 - 31-03-2016.pdf",
     "2016\\Comprovante BB - pagamento DARF Carnê-Leão - 31-03-2016.pdf"),
    ("2016\\DARF - IRPF cód. 0190 - 29-04-2016.pdf",
     "2016\\Comprovante BB - pagamento DARF Carnê-Leão - 29-04-2016.pdf"),
    ("2016\\DARF - IRPF cód. 0190 - 31-05-2016.pdf",
     "2016\\Comprovante BB - pagamento DARF Carnê-Leão - 31-05-2016.pdf"),
    ("2016\\Leão - cópia de segurança - IRPF 2016 ano-base 2015 - Maurício.dbk",
     "2016\\Leão 2016 - cópia de segurança - Maurício.dbk"),
    ("2016\\Leão - exportação - IRPF 2017 ano-base 2016 - Maurício.dec",
     "2016\\Leão 2016 - exportação - Maurício.dec"),
    ("2016\\Informe de rendimentos - Tribunal Superior Eleitoral - ano-base 2016 - Maurício Augusto Chiaramonte Vieira.pdf",
     "2016\\Informe de rendimentos - Tribunal Superior Eleitoral - ano-base 2016 - Maurício (2).pdf"),
    ("2017\\Declaração IRPF 2017 ano-base 2016 - Maurício.pdf",
     "2016\\Declaração IRPF 2017 ano-base 2016 - Maurício.pdf"),
    ("2017\\Recibo - entrega IRPF 2017 - Maurício.pdf",
     "2016\\Recibo - entrega IRPF 2017 - Maurício.pdf"),
    # ---- 2017 ----
    ("2017\\02533258113-IRPF-A-2018-2017-ORIGI.DBK",
     "2017\\Declaração IRPF 2018 ano-base 2017 - Maurício.dbk"),
    ("2017\\Informe de rendimentos - Itaú Unibanco S.A. - ano-base 2017 - Maurício.pdf",
     "2017\\Informe de rendimentos - Itaú Unibanco - ano-base 2017 - Maurício.pdf"),
    # ---- 2018 ----
    ("2018\\Relatório de inclusão negativo.pdf",
     "2018\\Relatório - Cadin Sisbacen não incluído - Receita Federal - Maurício - 15-04-2019.pdf"),
    # ---- 2019 ----
    ("2019\\FGTS.txt",
     "2019\\Nota - lembrete saque remanescente FGTS início de 2019.txt"),
    ("2019\\NF 2.pdf",
     "2019\\NF 67083 - radiologia odontológica - Clínica Fenelon - R$ 87,89 - 08-03-2019.pdf"),
    ("2019\\NF 6 - BOOK-32B45C8B6G.pdf",
     "2019\\NF 56294 - radiologia odontológica - Clínica Fenelon - R$ 67,20 - 06-08-2019.pdf"),
    ("2019\\RECIBO 1.pdf",
     "2019\\Recibo - entrega IRPF 2020 original - Maurício.pdf"),
    ("2019\\Recibo - entrega IRPF 2020 - Maurício.pdf",
     "2019\\Recibo - entrega IRPF 2020 retificadora 1 - Maurício.pdf"),
    ("2019\\Recibo - entrega IRPF 2020 - Maurício (2).pdf",
     "2019\\Recibo - entrega IRPF 2020 retificadora 2 - Maurício.pdf"),
    ("2019\\Recibo - entrega IRPF 2020 - Maurício.rec",
     "2019\\Recibo - entrega IRPF 2020 retificadora 2 - Maurício.rec"),
    ("2019\\Declaração IRPF 2020 ano-base 2019 - Maurício.dec",
     "2019\\Declaração IRPF 2020 ano-base 2019 - retificadora - Maurício.dec"),
    ("2019\\Relatório Cadin - Maurício.pdf",
     "2019\\Relatório - Cadin Sisbacen não incluído - Receita Federal - Maurício - 17-03-2020.pdf"),
    ("2019\\Informe de rendimentos - Itaú Unibanco S.A. - ano-base 2019 - Maurício.pdf",
     "2019\\Informe de rendimentos - Itaú Unibanco - ano-base 2019 - Maurício.pdf"),
    ("2019\\Informe de rendimentos - MercadoPago.com Representacoes Ltda - ano-base 2019 - Maurício.pdf",
     "2019\\Informe de rendimentos - Mercado Pago - ano-base 2019 - Maurício.pdf"),
    # ---- 2020 ----
    ("2020\\2_NFe.pdf",
     "2020\\Recibos - tratamento odontológico - Clínica Abreu Aires - 2x R$ 650,00 - 09 e 10-2020.pdf"),
    ("2020\\Declaração.pdf",
     "2020\\Declaração não identificada (arquivo corrompido).pdf"),
    ("2020\\Declaração IRPF 2020 ano-base 2019 - Maurício.pdf",
     "2019\\Declaração IRPF 2020 ano-base 2019 - retificadora - Maurício.pdf"),
    ("2020\\Recibo - entrega IRPF 2020 - Maurício.pdf",
     "2020\\Recibo - entrega IRPF 2021 - Maurício.pdf"),
    ("2020\\NF 000.068.012 - Hospital Pacini Ltda. - 07-2020.pdf",
     "2020\\NF 68012 - exames - Hospital Pacini - 07-2020.pdf"),
    ("2020\\Declaração de quitação anual - Banco GM S.A. - Contrato 6240228 - 2020.pdf",
     "2020\\Declaração de quitação anual - Banco GM - contrato 6240228 - 2020.pdf"),
    ("2020\\Informe de rendimentos - Banco GM S.A. - ano-base 2020 - Maurício.pdf",
     "2020\\Informe de rendimentos - Banco GM - ano-base 2020 - Maurício.pdf"),
    ("2020\\Informe de rendimentos - Itaú - ano-base 2020 - Maurício.pdf",
     "2020\\Informe de rendimentos - Itaú Unibanco - ano-base 2020 - Maurício.pdf"),
    ("2020\\Informe de rendimentos - MercadoPago.com Representações Ltda - ano-base 2020 - Maurício.pdf",
     "2020\\Informe de rendimentos - Mercado Pago - ano-base 2020 - Maurício.pdf"),
    ("2020\\Informe de rendimentos - Nu Financeira S.A. - ano-base 2020 - Maurício.pdf",
     "2020\\Informe de rendimentos - Nu Financeira - ano-base 2020 - Maurício.pdf"),
    ("2020\\Informe de rendimentos - PicPay Serviços S.A. - ano-base 2020 - Maurício.pdf",
     "2020\\Informe de rendimentos - PicPay - ano-base 2020 - Maurício.pdf"),
    # ---- 2021 ----
    ("2021\\GMAC.pdf",
     "2021\\Documento GMAC (arquivo corrompido).pdf"),
    ("2021\\Informe de rendimentos - Mercadopago.com Representacoes Ltda - ano-base 2021 - Maurício.pdf",
     "2021\\Informe de rendimentos - Mercado Pago - ano-base 2021 - Maurício.pdf"),
    ("2021\\Ana Carolina\\FAQ Sicoob.pdf",
     "2021\\Ana Carolina\\Guia - como declarar informe Sicoob no IRPF (exemplo ilustrativo).pdf"),
    ("2021\\Ana Carolina\\HR-V Dezembro 2021.pdf",
     "2021\\Ana Carolina\\Tabela Fipe - Honda HR-V EX 1.8 - 12-2021.pdf"),
    ("2021\\Ana Carolina\\Informe de rendimentos - Nu Financeira S.A. - ano-base 2021 - Ana Carolina.pdf",
     "2021\\Ana Carolina\\Informe de rendimentos - Nu Financeira - ano-base 2021 - Ana Carolina.pdf"),
    # ---- 2022 ----
    ("2022\\Recibo de entrega.jpg",
     "2022\\Comprovante - envio da declaração IRPF 2023 - Maurício - 15-05-2023.jpg"),
    ("2022\\Tabela Fipe - Chevrolet Cruze Sport LTZ 2018 - 12-2020.pdf",
     "2020\\Tabela Fipe - Chevrolet Cruze Sport LTZ 2018 - 12-2020.pdf"),
    ("2022\\Tabela Fipe - Chevrolet Cruze Sport LTZ 2018 - 12-2021.pdf",
     "2021\\Tabela Fipe - Chevrolet Cruze Sport LTZ 2018 - 12-2021.pdf"),
    # ---- 2023 ----
    ("2023\\Consórcio - senha 02570670.pdf",
     "2023\\Demonstrativo do consorciado - Consórcio imóvel Porto Seguro - Maurício (senha 02570670).pdf"),
    ("2023\\Declaração IRPF 2023 ano-base 2022 - Ana Carolina.pdf",
     "2022\\Ana Carolina\\Declaração IRPF 2023 ano-base 2022 - Ana Carolina (2).pdf"),
    ("2023\\Declaração IRPF 2023 ano-base 2022 - Maurício.pdf",
     "2022\\Declaração IRPF 2023 ano-base 2022 - Maurício (2).pdf"),
    ("2023\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina.pdf",
     "2023\\Ana Carolina\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina.pdf"),
    ("2023\\Declaração IRPF 2024 ano-base 2023 - Mauricio - Maurício.pdf",
     "2023\\Declaração IRPF 2024 ano-base 2023 - Maurício.pdf"),
    ("2023\\Recibo - entrega IRPF 2024 - Ana Carolina Miguel Meira e Silva.pdf",
     "2023\\Ana Carolina\\Recibo - entrega IRPF 2024 retificadora - Ana Carolina - 06-04-2026.pdf"),
    ("2023\\Recibo - entrega IRPF 2024 - Ana Carolina.pdf",
     "2023\\Ana Carolina\\Recibo - entrega IRPF 2024 retificadora - Ana Carolina - 25-05-2024.pdf"),
    ("2023\\Tabela Fipe Cruze - dez 2023 - R$ 95.388 - IRPF 2023.pdf",
     "2023\\Tabela Fipe - Chevrolet Cruze Sport LTZ 2018 - 12-2023.pdf"),
    ("2023\\Informe de rendimentos - Itaú Unibanco S.A. - ano-base 2023 - Maurício.pdf",
     "2023\\Informe de rendimentos - Itaú Unibanco - ano-base 2023 - Maurício.pdf"),
    ("2023\\Informe de rendimentos - Banco Itaucard S.A. - ano-base 2023 - Maurício.pdf",
     "2023\\Informe de rendimentos - Banco Itaucard - ano-base 2023 - Maurício.pdf"),
    ("2023\\Informe de rendimentos - BRB Banco de Brasília S A - ano-base 2023 - Maurício.pdf",
     "2023\\Informe de rendimentos - BRB - ano-base 2023 - Maurício.pdf"),
    # ---- 2024 (após rename Carol -> Ana Carolina) ----
    ("2024\\DARF - Maurício - IRPF 1ª parcela - R$ 143,29 - 30-05-2025.pdf",
     "2024\\DARF - IRPF 2025 1ª quota - Maurício - R$ 143,29 - 30-05-2025.pdf"),
    ("2024\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina (2).pdf",
     "2023\\Ana Carolina\\Declaração IRPF 2024 ano-base 2023 - retificadora - Ana Carolina.pdf"),
    ("2024\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina.pdf",
     "2023\\Ana Carolina\\Declaração IRPF 2024 ano-base 2023 com recibo - Ana Carolina.pdf"),
    ("2024\\Ana Carolina\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina.pdf",
     "2023\\Ana Carolina\\Declaração IRPF 2024 ano-base 2023 - Ana Carolina (2).pdf"),
    ("2024\\Declaração IRPF 2024 ano-base 2023 - Maurício.pdf",
     "2023\\Declaração IRPF 2024 ano-base 2023 - Maurício (2).pdf"),
    ("2024\\Declaração IRPF 2025 ano-base 2024 - Maurício (2).pdf",
     "2024\\Ficha Bens e Direitos (extrato) - IRPF 2025 - Maurício.pdf"),
    ("2024\\Informe de rendimentos - Porto Seguro Administradora de Consórcios Ltda - ano-base 2023 - Maurício.pdf",
     "2023\\Informe de rendimentos - Porto Seguro Consórcios - ano-base 2023 - Maurício.pdf"),
    ("2024\\Informe de rendimentos - Porto Seguro Administradora de Consórcios LTDA. - ano-base 2024 - Maurício.pdf",
     "2024\\Informe de rendimentos - Porto Seguro Consórcios - ano-base 2024 - Maurício.pdf"),
    ("2024\\Recibo - entrega IRPF 2024 - Maurício.png",
     "2023\\Recibo - entrega IRPF 2024 - Maurício.png"),
    ("2024\\Ana Carolina\\Recibo - entrega IRPF 2024 - Ana Carolina.pdf",
     "2024\\Ana Carolina\\Recibo - entrega IRPF 2025 - Ana Carolina.pdf"),
    ("2024\\Recibo - entrega IRPF 2025 - Ana Carolina Miguel Meira e Silva.pdf",
     "2024\\Ana Carolina\\Recibo - entrega IRPF 2025 - Ana Carolina (2).pdf"),
    ("2024\\Ana Carolina\\BB - dividendos.pdf",
     "2024\\Ana Carolina\\Informe - dividendos - Banco do Brasil - Ana Carolina (protegido por senha).pdf"),
    ("2024\\Ana Carolina\\Comprovante - rendimentos - Shirley - 2024 - 2025.pdf",
     "2024\\Ana Carolina\\Informe de rendimentos - Meira Clínica Médica - ano-base 2024 - Ana Carolina.pdf"),
    ("2024\\Ana Carolina\\Sicoob (2).jpeg",
     "2024\\Ana Carolina\\Extrato - conta capital Sicoob Credicom - 12-2024 - Ana Carolina.jpeg"),
    ("2024\\Ana Carolina\\Informe de rendimentos - XP Investimentos CCTVM S A; Banco XP S A - ano-base 2024 - Ana Carolina.pdf",
     "2024\\Ana Carolina\\Informe de rendimentos - XP Investimentos e Banco XP - ano-base 2024 - Ana Carolina.pdf"),
    ("2024\\Ana Carolina\\Informe de rendimentos - BRB Banco de Brasília S A - ano-base 2024 - Ana Carolina.pdf",
     "2024\\Ana Carolina\\Informe de rendimentos - BRB - ano-base 2024 - Ana Carolina.pdf"),
    ("2024\\Informe de rendimentos - BRB Banco de Brasília S A - ano-base 2024 - Maurício.pdf",
     "2024\\Informe de rendimentos - BRB - ano-base 2024 - Maurício.pdf"),
    ("2024\\Informe de rendimentos - Itaú Unibanco S.A. - ano-base 2024 - Maurício.pdf",
     "2024\\Informe de rendimentos - Itaú Unibanco - ano-base 2024 - Maurício.pdf"),
    ("2024\\Informe de rendimentos - Genial Investimentos Corretora de Valores Mobiliários S.A. - ano-base 2024 - Maurício.pdf",
     "2024\\Informe de rendimentos - Genial Investimentos - ano-base 2024 - Maurício.pdf"),
    # ---- 2025 (após rename Carol -> Ana Carolina) ----
    ("2025\\Ana Carolina\\1ª alteração e consolidação contratual - Junta Comercial DF (IRPF 2025 - Carol).pdf",
     "2025\\Ana Carolina\\Alteração contratual 1ª e consolidação - Meira Clínica Médica - Junta Comercial DF.pdf"),
    ("2025\\Ana Carolina\\2º Alteração contratual empresarial.pdf",
     "2025\\Ana Carolina\\Alteração contratual 2ª e consolidação - Meira Clínica Médica - Junta Comercial DF.pdf"),
    ("2025\\Ana Carolina\\DARF - Carol - IRPF 1ª Quota ou Quota única - R$ 368,58 - 29-05-2026.pdf",
     "2025\\Ana Carolina\\DARF - IRPF 2026 1ª quota - Ana Carolina - R$ 368,58 - 29-05-2026.pdf"),
    ("2025\\Ana Carolina\\Declaração IRPF 2025 ano-base 2024 - Ana Carolina.pdf",
     "2024\\Ana Carolina\\Declaração IRPF 2025 ano-base 2024 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Declaração IRPF 2026 ano-base 2025 - Gabriel Afonso Miguel Meira e Silva.pdf",
     "2025\\Gabriel\\Declaração IRPF 2026 ano-base 2025 - Gabriel.pdf"),
    ("2025\\Ana Carolina\\Informe de rendimentos - Carol - INSTITUTO DE GESTAO - ano-base 2025 - Ana Carolina.pdf",
     "2025\\Ana Carolina\\Informe de rendimentos - Instituto de Gestão - ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Informe de rendimentos - Carol - Meira Clínica Médica - ano-base 2025 - Ana Carolina.pdf",
     "2025\\Ana Carolina\\Informe de rendimentos - Meira Clínica Médica - ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Informe de rendimentos - Carol - Nubank - ano-base 2025 - Ana Carolina.pdf",
     "2025\\Ana Carolina\\Informe de rendimentos - Nubank - ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Informe de rendimentos - SICOOB CREDICOM - ano-base 2025 - Ana Carolina.pdf",
     "2025\\Ana Carolina\\Informe de rendimentos - Sicoob Credicom - ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Informe de rendimentos - BRB Banco de Brasília S A - ano-base 2025 - Ana Carolina.pdf",
     "2025\\Ana Carolina\\Informe de rendimentos - BRB - ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\Ana Carolina\\Relatório auxiliar para declaração de IRPF 2025.pdf",
     "2025\\Ana Carolina\\Relatório myProfit - auxiliar IRPF 2026 ano-base 2025 - Ana Carolina.pdf"),
    ("2025\\DARF - Maurício - IRPF 1ª quota ou quota única - R$ 634,27 - 29-05-2026.pdf",
     "2025\\DARF - IRPF 2026 1ª quota - Maurício - R$ 634,27 - 29-05-2026.pdf"),
    ("2025\\DARF - IRPF 1ª quota ou quota única - R$ 634,27 - 29-05-2026.pdf",
     "2025\\DARF - IRPF 2026 1ª quota - Maurício - R$ 634,27 - 29-05-2026 (2).pdf"),
    ("2025\\Informe de rendimentos - Nomad- ano-base 2025 - Maurício.pdf",
     "2025\\Guia - como declarar Nomad no IRPF - ano-base 2025 - Maurício.pdf"),
    ("2025\\Recibo - Mauricio - entrega IRPF 2026 - Maurício.pdf",
     "2025\\Recibo - entrega IRPF 2026 - Maurício.pdf"),
    ("2025\\Informe de rendimentos - Genial Institucional Corretora de Câmbio, Títulos e Valores Mobiliários S.A. - ano-base 2025 - Maurício.pdf",
     "2025\\Informe de rendimentos - Genial Institucional - ano-base 2025 - Maurício.pdf"),
    ("2025\\Informe de rendimentos - Genial Investimentos Corretora de Valores Mobiliários S.A. - ano-base 2025 - Maurício.pdf",
     "2025\\Informe de rendimentos - Genial Investimentos - ano-base 2025 - Maurício.pdf"),
    ("2025\\Informe de rendimentos - Itaú Unibanco S.A. - ano-base 2025 - Maurício.pdf",
     "2025\\Informe de rendimentos - Itaú Unibanco - ano-base 2025 - Maurício.pdf"),
    ("2025\\Informe de rendimentos - Mercado Pago Instituição de Pagamento Ltda - ano-base 2025 - Maurício.pdf",
     "2025\\Informe de rendimentos - Mercado Pago - ano-base 2025 - Maurício.pdf"),
    ("2025\\Informe de rendimentos - BRB Banco de Brasília S A - ano-base 2025 - Maurício.pdf",
     "2025\\Informe de rendimentos - BRB - ano-base 2025 - Maurício.pdf"),
    # ---- raiz ----
    ("Declaração DEFIS exercício 2025 ano-calendário 2024 - Meira Clínica Médica LTDA.pdf",
     "2024\\Ana Carolina\\DEFIS 2025 ano-calendário 2024 - Meira Clínica Médica.pdf"),
    ("Declaração DEFIS exercício 2026 ano-calendário 2025 - Meira Clínica Médica LTDA.pdf",
     "2025\\Ana Carolina\\DEFIS 2026 ano-calendário 2025 - Meira Clínica Médica.pdf"),
    ("Recibo - entrega da DEFIS 2024 - Meira Clínica Médica LTDA - 13-03-2025.pdf",
     "2024\\Ana Carolina\\Recibo - entrega DEFIS 2025 ano-calendário 2024 - Meira Clínica Médica - 13-03-2025.pdf"),
    # ---- Documentos antigos ----
    ("Documentos antigos\\IRPF - Moisés.pdf",
     "Documentos antigos\\Recibo - entrega IRPF 2015 - Moisés de Paula Bernardes.pdf"),
    ("Documentos antigos\\IRPF - Moisés 2.pdf",
     "Documentos antigos\\Declaração IRPF 2015 ano-base 2014 - Moisés de Paula Bernardes.pdf"),
    ("Documentos antigos\\Termo de adesão ao DTE - TABELIONATO DE PROTESTO DE TITULO.pdf",
     "Documentos antigos\\Termo de adesão ao DTE - Tabelionato de Protesto de Títulos.pdf"),
]


def pre(path: str) -> str:
    return "\\\\?\\" + path if not path.startswith("\\\\?\\") else path


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def main() -> None:
    apply = "--apply" in sys.argv
    rows = []
    erros = 0

    def do(kind: str, src_rel: str, dst_rel: str) -> None:
        nonlocal erros
        src = os.path.join(BASE, nfc(src_rel))
        dst = os.path.join(BASE, nfc(dst_rel))
        if len(dst) > 240:
            print(f"  [ERRO caminho>240] {dst_rel}")
            erros += 1
            return
        if kind != "mkdir" and not os.path.exists(pre(src)):
            # em dry-run, pastas Carol ainda não foram renomeadas p/ Ana Carolina
            alt = src.replace("\\Ana Carolina\\", "\\Carol\\") if not apply else src
            if apply or not os.path.exists(pre(alt)):
                print(f"  [FALTA ORIGEM] {src_rel}")
                erros += 1
                rows.append((kind, src_rel, dst_rel, "origem ausente"))
                return
        if kind == "mkdir":
            if apply:
                os.makedirs(pre(dst), exist_ok=True)
            rows.append((kind, "", dst_rel, "ok"))
            return
        final = dst
        if os.path.exists(pre(dst)) and os.path.normcase(src) != os.path.normcase(dst):
            stem, ext = os.path.splitext(dst)
            n = 2
            while os.path.exists(pre(f"{stem} ({n}){ext}")):
                n += 1
            final = f"{stem} ({n}){ext}"
            print(f"  [CONFLITO -> sufixo ({n})] {dst_rel}")
        if apply:
            os.makedirs(pre(os.path.dirname(final)), exist_ok=True)
            os.rename(pre(src), pre(final))
        rows.append((kind, src_rel, os.path.relpath(final, BASE), "ok"))

    print(("APLICANDO" if apply else "DRY-RUN") + f" — base: {BASE}\n")
    for d in MKDIRS:
        do("mkdir", "", d)
    for src, dst in DIR_RENAMES:
        do("movedir", src, dst)
    for src, dst in OPS:
        do("move" if os.path.dirname(src) != os.path.dirname(dst) else "rename", src, dst)

    ok = sum(1 for r in rows if r[3] == "ok")
    print(f"\n{ok} operações ok | {erros} erros/faltas")
    if apply:
        with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["timestamp", "acao", "origem", "destino", "status"])
            ts = datetime.now().isoformat(timespec="seconds")
            for r in rows:
                w.writerow([ts, *r])
        print(f"Log de reversão: {LOG}")


if __name__ == "__main__":
    main()
