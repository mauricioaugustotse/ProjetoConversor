import argparse
import csv
import os
import tempfile
import time
from pathlib import Path


DEFAULT_CSV_PATH = Path(r"C:\Users\mauri\ProjetoConversor\boletins_de_jurisprudencia_TRF1.csv")


def _gerar_saida_alternativa(csv_path: Path) -> Path:
    candidato = csv_path.with_name(f"{csv_path.stem}_limpo{csv_path.suffix}")
    contador = 2
    while candidato.exists():
        candidato = csv_path.with_name(f"{csv_path.stem}_limpo_{contador}{csv_path.suffix}")
        contador += 1
    return candidato


def _mover_com_fallback(caminho_temp: Path, csv_path: Path) -> Path:
    for tentativa in range(5):
        try:
            os.replace(caminho_temp, csv_path)
            return csv_path
        except PermissionError:
            if tentativa < 4:
                time.sleep(0.5)

    saida_alternativa = _gerar_saida_alternativa(csv_path)
    os.replace(caminho_temp, saida_alternativa)
    print(
        f'Aviso: não foi possível sobrescrever "{csv_path}" (arquivo em uso). '
        f'Arquivo salvo em "{saida_alternativa}".'
    )
    return saida_alternativa


def limpar_coluna(csv_path: Path, nome_coluna: str = "noticia") -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as arquivo_entrada:
        amostra = arquivo_entrada.read(4096)
        arquivo_entrada.seek(0)

        try:
            dialecto = csv.Sniffer().sniff(amostra)
        except csv.Error:
            dialecto = csv.excel

        leitor = csv.DictReader(arquivo_entrada, dialect=dialecto, restkey="__extra__")
        if not leitor.fieldnames:
            raise ValueError("O CSV não possui cabeçalho.")
        if nome_coluna not in leitor.fieldnames:
            raise ValueError(f'A coluna "{nome_coluna}" não existe no arquivo.')

        fd_temp, caminho_temp = tempfile.mkstemp(
            dir=str(csv_path.parent),
            prefix=f"{csv_path.stem}_tmp_",
            suffix=".csv",
        )
        os.close(fd_temp)

        try:
            with open(caminho_temp, "w", encoding="utf-8-sig", newline="") as arquivo_saida:
                escritor = csv.DictWriter(
                    arquivo_saida,
                    fieldnames=leitor.fieldnames,
                    delimiter=getattr(dialecto, "delimiter", ","),
                    quotechar=getattr(dialecto, "quotechar", '"') or '"',
                    lineterminator=getattr(dialecto, "lineterminator", "\n") or "\n",
                    quoting=csv.QUOTE_MINIMAL,
                    doublequote=True,
                    extrasaction="ignore",
                )
                escritor.writeheader()

                for linha in leitor:
                    linha.pop("__extra__", None)
                    linha[nome_coluna] = ""
                    escritor.writerow(linha)

            return _mover_com_fallback(Path(caminho_temp), csv_path)
        except Exception:
            if os.path.exists(caminho_temp):
                os.remove(caminho_temp)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Apaga todo o conteúdo da coluna "noticia" em um arquivo CSV.'
    )
    parser.add_argument(
        "arquivo_csv",
        nargs="?",
        default=str(DEFAULT_CSV_PATH),
        help=f"Caminho do CSV (padrão: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--coluna",
        default="noticia",
        help='Nome da coluna a ser limpa (padrão: "noticia")',
    )
    args = parser.parse_args()

    caminho = Path(args.arquivo_csv)
    destino = limpar_coluna(caminho, args.coluna)
    print(f'Coluna "{args.coluna}" limpa com sucesso em: {destino}')


if __name__ == "__main__":
    main()
