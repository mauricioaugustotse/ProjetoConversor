# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import csv
import os
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
# --------------------------------------------------------------------

fitz = None
openai = None
tqdm_asyncio = None

try:
    import fitz  # PyMuPDF
except ImportError:
    pass

try:
    import openai
except ImportError:
    pass

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    pass

try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except ImportError:
    print("dotenv não instalado, pulando o carregamento de variáveis de .env")
except Exception as e:
    print(f"Erro ao carregar .env: {e}")

missing_obrigatorias = []
missing_opcionais = []
if fitz is None:
    missing_obrigatorias.append("pymupdf")
if openai is None:
    missing_opcionais.append("openai")
if tqdm_asyncio is None:
    missing_opcionais.append("tqdm")

if missing_obrigatorias:
    print(f"{C.RED}{C.BOLD}ERRO: Dependência obrigatória não encontrada: {', '.join(missing_obrigatorias)}.{C.END}")
    print(f"{C.YELLOW}Instale com: python -m pip install {' '.join(missing_obrigatorias)}{C.END}")
if missing_opcionais:
    print(f"{C.YELLOW}AVISO: Dependências opcionais ausentes: {', '.join(missing_opcionais)}.{C.END}")
    print(f"{C.YELLOW}Para habilitar a classificação por IA: python -m pip install {' '.join(missing_opcionais)}{C.END}")



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODELO_IA = "gpt-5-mini"

CATEGORIAS_PERMITIDAS = [
    "Casa", "Educação", "Eletrônicos", "Lazer", "Outros", "Restaurante",
    "Saúde", "Serviços", "Supermercado", "Transporte", "Vestuário", "Viagem"
]

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5
MAX_CONCURRENT_REQUESTS = 10

async def classificar_transacao(transacao: Dict[str, Any], cliente: Any) -> str:
    """Usa a API da OpenAI para classificar a categoria de uma única transação."""
    descricao = transacao['descricao']
    prompt = f"""
    Classifique a seguinte descrição de despesa de cartão de crédito em uma das categorias abaixo.
    Responda APENAS com o nome exato da categoria.

    Categorias Válidas: {', '.join(CATEGORIAS_PERMITIDAS)}

    Descrição da Despesa: "{descricao}"

    Categoria:
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await cliente.chat.completions.create(
                model=MODELO_IA,
                messages=[
                    {"role": "system", "content": "Você é um assistente de finanças pessoais preciso e eficiente."},
                    {"role": "user", "content": prompt}
                ],
            )
            categoria = response.choices[0].message.content.strip().replace('"', '').replace('.', '')
            if categoria in CATEGORIAS_PERMITIDAS:
                return categoria
            else:
                return "Outros"
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            delay = INITIAL_RETRY_DELAY * (2 ** attempt)
            print(f"{C.YELLOW}AVISO: Erro de API para '{descricao}' (tentativa {attempt + 1}/{MAX_RETRIES}). Tentando novamente em {delay}s...{C.END}")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"{C.RED}ERRO INESPERADO na API para '{descricao}': {e}{C.END}")
            return "Outros"

    return "Outros"

async def processar_categorias_em_lote(transacoes_por_cartao: Dict[str, List[Dict[str, Any]]]):
    """Orquestra a classificação de todas as transações de forma concorrente."""
    if openai is None:
        print(f"{C.YELLOW}Classificação por IA indisponível: biblioteca 'openai' não instalada.{C.END}")
        return
    if not OPENAI_API_KEY:
        print(f"\n{C.RED}{C.BOLD}ERRO CRÍTICO: A variável OPENAI_API_KEY não foi configurada.{C.END}")
        print(f"{C.YELLOW}Defina OPENAI_API_KEY no ambiente antes de executar este script.{C.END}")
        return

    cliente_openai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    tarefas_para_ia = []
    for cartao, transacoes in transacoes_por_cartao.items():
        for transacao in transacoes:
            if float(transacao['valor']) < 0:
                tarefas_para_ia.append(transacao)

    if not tarefas_para_ia:
        print(f"{C.YELLOW}Nenhuma despesa encontrada para ser classificada.{C.END}")
        return
        
    print(f"\n{C.CYAN}{C.BOLD}🤖 Iniciando classificação de {len(tarefas_para_ia)} despesas com IA (usando {MODELO_IA})...{C.END}")
    print(f"{C.CYAN}   Isso pode levar alguns minutos. Serão feitas até {MAX_CONCURRENT_REQUESTS} chamadas simultâneas à API.{C.END}")

    tasks = [classificar_transacao(transacao, cliente_openai) for transacao in tarefas_para_ia]
    if tqdm_asyncio:
        resultados_categorias = await tqdm_asyncio.gather(*tasks, desc="Classificando despesas")
    else:
        resultados_categorias = await asyncio.gather(*tasks)
    
    for transacao, categoria in zip(tarefas_para_ia, resultados_categorias):
        transacao['categoria'] = categoria
        
    print(f"{C.GREEN}✓ Classificação por IA concluída.{C.END}")
# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def selecionar_arquivo_pdf() -> Optional[Path]:
    """Abre uma janela para selecionar o PDF; faz fallback para busca local se não houver GUI."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"{C.YELLOW}AVISO: Não foi possível abrir a janela de seleção ({e}).{C.END}")
        return encontrar_arquivo_pdf()

    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo PDF",
            filetypes=[("Arquivos PDF", "*.pdf"), ("Todos os arquivos", "*.*")]
        )
        root.destroy()
    except Exception as e:
        print(f"{C.YELLOW}AVISO: Erro ao abrir a janela de seleção ({e}).{C.END}")
        return encontrar_arquivo_pdf()

    if not file_path:
        print(f"{C.YELLOW}Nenhum arquivo selecionado.{C.END}")
        return None

    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"{C.RED}ERRO: O arquivo selecionado não é um PDF.{C.END}")
        return None

    print(f"{C.GREEN}✓ Arquivo PDF selecionado: {pdf_path}{C.END}")
    return pdf_path

def encontrar_arquivo_pdf() -> Optional[Path]:
    print(f"{C.CYAN}🔎 Procurando por arquivo .pdf na pasta atual...{C.END}")
    pdfs = list(Path.cwd().glob("*.pdf"))
    if not pdfs:
        print(f"{C.RED}ERRO: Nenhum arquivo PDF encontrado no diretório: {Path.cwd()}{C.END}")
        return None
    pdf_path = pdfs[0]
    if len(pdfs) > 1:
        print(f"{C.YELLOW}AVISO: Múltiplos arquivos PDF encontrados. Usando o primeiro: {pdf_path.name}{C.END}")
    else:
        print(f"{C.GREEN}✓ Arquivo PDF encontrado: {pdf_path.name}{C.END}")
    return pdf_path

def extrair_texto_do_pdf(pdf_path: Path) -> str:
    print(f"{C.CYAN}📄 Extraindo texto do arquivo PDF...{C.END}")
    if fitz is None:
        print(f"{C.RED}ERRO: PyMuPDF (pymupdf) não está instalado. Não é possível ler o PDF.{C.END}")
        return ""
    texto_completo = ""
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                texto_completo += page.get_text("text")
                print(f"   Lendo página {i+1}/{len(doc)}...")
        print(f"{C.GREEN}✓ Extração de texto concluída com sucesso.{C.END}")
        return texto_completo
    except Exception as e:
        print(f"{C.RED}ERRO ao tentar ler o arquivo PDF: {e}{C.END}")
        return ""

def extrair_data_vencimento(texto_pdf: str) -> tuple[int, int]:
    print(f"{C.CYAN}🗓️ Identificando data de vencimento da fatura...{C.END}")
    match = re.search(r"vencimento\s+em:\s+\d{2}/(\d{2})/(\d{4})", texto_pdf, re.IGNORECASE)
    if match:
        mes = int(match.group(1)); ano = int(match.group(2))
        print(f"{C.GREEN}✓ Vencimento identificado: Mês {mes}, Ano {ano}.{C.END}")
        return mes, ano
    print(f"{C.YELLOW}AVISO: Não foi possível encontrar a data de vencimento. Usando data atual como fallback.{C.END}")
    hoje = datetime.now()
    return hoje.month, hoje.year

def _limpar_descricao(desc: str) -> str:
    d = re.sub(r'\s+', ' ', desc).strip()
    # remove frases de metadado caso venham coladas na linha
    d = re.sub(r'Parcela\s+Lojista\s+Visa\s+Parc\.?\s*\d{1,2}\s*/\s*\d{1,2}', '', d, flags=re.IGNORECASE)
    d = re.sub(r'Compra\s+a\s+Vista', '', d, flags=re.IGNORECASE)
    d = re.sub(r'\bPARC\.?\s*\d{1,2}\s*/\s*\d{1,2}\b', '', d, flags=re.IGNORECASE)
    d = re.sub(r'IOF\s+Transacoes\s+Exterior\s*R\$', '', d, flags=re.IGNORECASE)
    d = re.sub(r'Compra\s+Exterior\s*R\$\s*-\s*Visa', '', d, flags=re.IGNORECASE)
    d = re.sub(r'\s{2,}', ' ', d).strip(' -')
    return d

def _append_frac(base: str, frac: Optional[str]) -> str:
    if not frac:
        return base
    if re.search(r'\(\s*\d{1,2}\s*/\s*\d{1,2}\s*\)\s*$', base):
        return base
    return f"{base} ({frac})"

# ==============================================================================
# PARSER COM ALINHAMENTO DE METADADOS ("PARCELA LOJISTA..." / "COMPRA A VISTA")
# ==============================================================================

def parsear_transacoes(pdf_path: Path, mes_venc: int, ano_venc: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Motor com associação estável entre metadados e linhas de transação:
    • Varre por página, preservando a ordem de aparição.
    • Alinha cada metadado ("Parcela Lojista Visa Parc.x/y", "Compra a Vista", "IOF Transacoes Exterior R$",
      "Compra Exterior R$ - Visa") à PRÓXIMA linha de transação pendente.
    • Se o metadado vier ANTES das linhas (como ocorre em alguns blocos do PDF), ele é enfileirado
      e aplicado à primeira linha que surgir (uma por uma).
    • Se vier DEPOIS das linhas, consumirá a mais antiga pendente (o que mantém o pareamento na ordem).
    """
    print(f"{C.CYAN}💳 Extraindo transações com alinhamento de metadados por página...{C.END}")
    if fitz is None:
        print(f"{C.RED}ERRO: PyMuPDF (pymupdf) não está instalado. Não é possível processar o PDF.{C.END}")
        return {}

    transacoes_por_cartao: Dict[str, List[Dict[str, Any]]] = {}

    def inferir_ano(mes_tx: int) -> int:
        return ano_venc if mes_tx <= mes_venc else (ano_venc - 1)

    try:
        with fitz.open(pdf_path) as doc:
            pat_card = re.compile(r"(\d{4}\.\*{4}\.\*{4}\.\d{4})")
            pat_tx   = re.compile(r"(?m)^(\d{2}/\d{2})\s+([\s\S]+?)\s*(R\$\s*[\d\.,]+)([\-\+])")
            pat_mk_parc = re.compile(r"Parcela\s+Lojista\s+Visa\s+Parc\.?\s*(\d{1,2}\s*/\s*\d{1,2})", re.IGNORECASE)
            pat_mk_vista = re.compile(r"Compra\s+a\s+Vista", re.IGNORECASE)
            pat_mk_ignore = re.compile(r"IOF\s+Transacoes\s+Exterior\s*R\$|Compra\s+Exterior\s*R\$\s*-\s*Visa", re.IGNORECASE)

            current_card_global: Optional[str] = None

            for page in doc:
                texto_pagina = page.get_text("text")

                tokens = []
                tokens += [('card', m.start(), m.group(1), m) for m in pat_card.finditer(texto_pagina)]
                tokens += [('tx',   m.start(), m, m) for m in pat_tx.finditer(texto_pagina)]
                tokens += [('mkp',  m.start(), m.group(1), m) for m in pat_mk_parc.finditer(texto_pagina)]
                tokens += [('mkv',  m.start(), None, m) for m in pat_mk_vista.finditer(texto_pagina)]
                tokens += [('mki',  m.start(), None, m) for m in pat_mk_ignore.finditer(texto_pagina)]
                tokens.sort(key=lambda x: x[1])

                current_card = current_card_global
                pendentes_idx: List[int] = []
                pagina_transacoes: List[Dict[str, Any]] = []
                atribuicoes: Dict[int, Optional[str]] = {}  # None => consumiu metadado sem fração
                queue_markers: deque[Tuple[str, Optional[str]]] = deque()

                for typ, pos, content, mobj in tokens:
                    if typ == 'card':
                        current_card = content
                        current_card_global = content
                        continue

                    if typ == 'tx':
                        data_str, desc_raw, valor_raw, sinal = mobj.groups()
                        dia_str, mes_str = data_str.split('/')
                        ano_tx = inferir_ano(int(mes_str))

                        valor_txt = valor_raw.replace("R$", "").strip().replace(".", "").replace(",", ".")
                        try:
                            valor_num = float(valor_txt)
                        except Exception:
                            continue
                        if sinal == '+':
                            valor_num = -valor_num
                        valor_final = str(int(valor_num)) if valor_num == int(valor_num) else f"{valor_num:.2f}"

                        # Captura fração inline "PARC.x/y" antes da limpeza
                        m_local_frac = re.search(r'\bPARC\.?\s*(\d{1,2}\s*/\s*\d{1,2})\b', desc_raw, flags=re.IGNORECASE)
                        local_frac = re.sub(r'\s+', '', m_local_frac.group(1)) if m_local_frac else None
                        descricao_base = _limpar_descricao(desc_raw)

                        item = {
                            "data": f"{dia_str}/{mes_str}/{ano_tx}",
                            "descricao": descricao_base,
                            "valor": valor_final,
                            "conta": "BRB",
                            "categoria": "A classificar",
                            "_card": current_card,
                        }
                        idx = len(pagina_transacoes)
                        pagina_transacoes.append(item)

                        # Se há metadados enfileirados (vieram antes), consome 1 para esta linha
                        if queue_markers:
                            mk_typ, mk_frac = queue_markers.popleft()
                            atribuicoes[idx] = (re.sub(r'\s+', '', mk_frac) if mk_frac else None) if mk_typ == 'mkp' else None
                        else:
                            pendentes_idx.append(idx)
                        continue

                    # metadados: aplica ao próximo pendente; se não houver, enfileira
                    if typ in ('mkp', 'mkv', 'mki'):
                        frac = re.sub(r'\s+', '', content) if typ == 'mkp' else None
                        if pendentes_idx:
                            alvo = pendentes_idx.pop(0)
                            atribuicoes[alvo] = frac  # None para 'mkv'/'mki' => apenas consumir alinhamento
                        else:
                            queue_markers.append((typ, frac))

                # aplica as frações atribuídas
                for i, tx in enumerate(pagina_transacoes):
                    frac = atribuicoes.get(i, None)
                    # aplica fração vinda do metadado e/ou a fração inline (PARC.x/y)
                    if frac:
                        tx['descricao'] = _append_frac(tx['descricao'], frac)
                    if 'local_frac' in locals() and local_frac:
                        tx['descricao'] = _append_frac(tx['descricao'], local_frac)

                    card_key = tx.pop("_card", None)
                    transacoes_por_cartao.setdefault(card_key, []).append(tx)

        total = sum(len(v) for v in transacoes_por_cartao.values())
        print(f"{C.GREEN}✓ Análise concluída. {total} transações extraídas.{C.END}")
        return transacoes_por_cartao

    except Exception as e:
        print(f"{C.RED}ERRO durante a extração: {e}{C.END}")
        return {}

# ==============================================================================
# IA (Opcional) — Classificação de categoria
# ==============================================================================

async def classificar_transacao(transacao: Dict[str, Any], cliente) -> str:
    descricao = transacao['descricao']
    prompt = f"""
    Classifique a seguinte descrição de despesa de cartão de crédito em uma das categorias abaixo.
    Responda APENAS com o nome exato da categoria.

    Categorias Válidas: {', '.join(CATEGORIAS_PERMITIDAS)}

    Descrição da Despesa: "{descricao}"

    Categoria:
    """
    if not cliente:
        return "Outros"

    for attempt in range(MAX_RETRIES):
        try:
            response = await cliente.chat.completions.create(
                model=MODELO_IA,
                messages=[
                    {"role": "system", "content": "Você é um assistente de finanças pessoais preciso e eficiente."},
                    {"role": "user", "content": prompt}
                ],
            )
            categoria = response.choices[0].message.content.strip().replace('"', '').replace('.', '')
            return categoria if categoria in CATEGORIAS_PERMITIDAS else "Outros"
        except Exception:
            delay = INITIAL_RETRY_DELAY * (2 ** attempt)
            print(f"{C.YELLOW}AVISO: Erro de API para '{descricao}' (tentativa {attempt + 1}/{MAX_RETRIES}). Tentando novamente em {delay}s...{C.END}")
            await asyncio.sleep(delay)
    return "Outros"



def salvar_csvs(
    transacoes_por_cartao: Dict[str, List[Dict[str, Any]]],
    base_name: str = "fatura_cartao_final",
    output_dir: Optional[Path] = None,
):
    print(f"\n{C.CYAN}{C.BOLD}💾 Salvando arquivos CSV...{C.END}")
    if not any(transacoes_por_cartao.values()):
        print(f"{C.YELLOW}Nenhuma transação para salvar.{C.END}")
        return
    destino = output_dir or Path.cwd()
    for cartao, transacoes in transacoes_por_cartao.items():
        if not transacoes:
            print(f"{C.YELLOW}AVISO: Sem dados para o cartão {cartao}.{C.END}")
            continue
        last4 = cartao.split('.')[-1] if cartao else 'None'
        nome_arquivo = f"{base_name}_{last4}.csv"
        try:
            caminho_saida = destino / nome_arquivo
            with open(caminho_saida, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
                writer.writerow(["Data", "Descrição", "Valor", "Conta", "Categoria"])
                for t in transacoes:
                    writer.writerow([t.get('data',''), t.get('descricao',''), t.get('valor',''), t.get('conta',''), t.get('categoria','')])
            print(f"{C.GREEN}✓ Arquivo salvo: {caminho_saida}{C.END}")
        except IOError as e:
            print(f"{C.RED}ERRO ao salvar {nome_arquivo}: {e}{C.END}")

# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    inicio = datetime.now()
    print(f"{C.BOLD}{C.BLUE}{'='*80}{C.END}")
    print(f"{C.BOLD}{C.BLUE}  EXTRATOR DE FATURA — Parcelas por Metadado v2.0{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*80}{C.END}\n")

    try:
        pdf_path = selecionar_arquivo_pdf()
        if not pdf_path:
            return

        texto_pdf = extrair_texto_do_pdf(pdf_path)
        if not texto_pdf:
            return

        mes_venc, ano_venc = extrair_data_vencimento(texto_pdf)
        transacoes_por_cartao = parsear_transacoes(pdf_path, mes_venc, ano_venc)

        await processar_categorias_em_lote(transacoes_por_cartao)
        salvar_csvs(transacoes_por_cartao, output_dir=pdf_path.parent)

    except Exception:
        print(f"\n{C.RED}{C.BOLD}{'!'*80}{C.END}")
        print(f"{C.RED}{C.BOLD}Ocorreu um erro inesperado durante a execução:{C.END}")
        print(f"{C.RED}{traceback.format_exc()}{C.END}")
        print(f"{C.RED}{C.BOLD}{'!'*80}{C.END}")
    finally:
        fim = datetime.now()
        print(f"\n{C.BOLD}{C.GREEN}Execução finalizada em: {fim - inicio}{C.END}")

if __name__ == "__main__":
    asyncio.run(main())
