
import argparse
import csv
import os
import sys
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt


def carregar_dados(arquivo_csv):
    """Lê os resultados das execuções do algoritmo genético."""

    dados = []

    with open(arquivo_csv, "r", newline="", encoding="utf-8-sig") as arquivo:
        leitor = csv.DictReader(arquivo)

        colunas_necessarias = {
            "instancia",
            "experimento",
            "rodada",
            "avaliacoes",
            "melhor_h",
            "media_populacao",
            "diversidade",
        }

        if not leitor.fieldnames:
            raise ValueError("O CSV está vazio ou não possui cabeçalho.")

        faltantes = colunas_necessarias - set(leitor.fieldnames)

        if faltantes:
            raise ValueError(
                f"Colunas ausentes no CSV: {', '.join(sorted(faltantes))}"
            )

        for linha in leitor:
            dados.append({
                "instancia": linha["instancia"].strip(),
                "experimento": linha["experimento"].strip(),
                "rodada": linha["rodada"].strip(),
                "avaliacoes": int(float(linha["avaliacoes"])),
                "melhor_h": float(linha["melhor_h"]),
                "media_populacao": float(linha["media_populacao"]),
                "diversidade": float(linha["diversidade"]),
            })

    if not dados:
        raise ValueError("O CSV não contém resultados.")

    return dados


def agregar_por_avaliacoes(dados, metrica):
    """
    Calcula a mediana entre as execuções para cada número de avaliações.
    Retorna uma série para cada combinação de instância e experimento.
    """

    grupos = defaultdict(list)

    for linha in dados:
        chave = (
            linha["instancia"],
            linha["experimento"],
            linha["avaliacoes"],
        )

        grupos[chave].append(linha[metrica])

    series = defaultdict(list)

    for (instancia, experimento, avaliacoes), valores in grupos.items():
        valor_mediano = median(valores)

        series[(instancia, experimento)].append(
            (avaliacoes, valor_mediano)
        )

    for chave in series:
        series[chave].sort(key=lambda ponto: ponto[0])

    return series


def salvar_grafico(caminho, titulo, eixo_x, eixo_y):
    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo: {caminho}")


def plotar_metrica(
    series,
    instancias,
    experimentos,
    pasta_saida,
    metrica,
    titulo_base,
    eixo_y,
    nome_arquivo,
):
    """Gera um gráfico da métrica para cada instância."""

    for instancia in instancias:
        plt.figure(figsize=(10, 6))

        for experimento in experimentos:
            pontos = series.get((instancia, experimento), [])

            if not pontos:
                continue

            x = [ponto[0] for ponto in pontos]
            y = [ponto[1] for ponto in pontos]

            plt.plot(
                x,
                y,
                label=experimento,
                linewidth=1.8,
            )

        salvar_grafico(
            os.path.join(
                pasta_saida,
                f"{nome_arquivo}_{instancia}.png",
            ),
            f"{titulo_base} — {instancia}",
            "Número de avaliações da função objetivo",
            eixo_y,
        )


def gerar_graficos(dados, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)

    instancias = sorted(
        {linha["instancia"] for linha in dados}
    )

    experimentos = sorted(
        {linha["experimento"] for linha in dados}
    )

    # 1. Convergência: melhor solução encontrada.
    series_melhor = agregar_por_avaliacoes(dados, "melhor_h")

    plotar_metrica(
        series=series_melhor,
        instancias=instancias,
        experimentos=experimentos,
        pasta_saida=pasta_saida,
        metrica="melhor_h",
        titulo_base="Convergência do algoritmo genético",
        eixo_y="Mediana do melhor custo h",
        nome_arquivo="convergencia",
    )

    # 2. Evolução da média da população.
    series_media = agregar_por_avaliacoes(
        dados, "media_populacao"
    )

    plotar_metrica(
        series=series_media,
        instancias=instancias,
        experimentos=experimentos,
        pasta_saida=pasta_saida,
        metrica="media_populacao",
        titulo_base="Evolução da média da população",
        eixo_y="Mediana da média do custo h da população",
        nome_arquivo="media_populacao",
    )

    # 3. Evolução da diversidade.
    series_diversidade = agregar_por_avaliacoes(
        dados, "diversidade"
    )

    plotar_metrica(
        series=series_diversidade,
        instancias=instancias,
        experimentos=experimentos,
        pasta_saida=pasta_saida,
        metrica="diversidade",
        titulo_base="Diversidade da população",
        eixo_y="Mediana do número de indivíduos distintos",
        nome_arquivo="diversidade",
    )

    # 4. Comparação entre melhor indivíduo e média da população.
    experimento = "OX" if "OX" in experimentos else experimentos[0]

    for instancia in instancias:
        pontos_melhor = series_melhor.get(
            (instancia, experimento), []
        )

        pontos_media = series_media.get(
            (instancia, experimento), []
        )

        if not pontos_melhor or not pontos_media:
            continue

        plt.figure(figsize=(10, 6))

        x_melhor = [p[0] for p in pontos_melhor]
        y_melhor = [p[1] for p in pontos_melhor]

        x_media = [p[0] for p in pontos_media]
        y_media = [p[1] for p in pontos_media]

        plt.plot(
            x_melhor,
            y_melhor,
            label="Mediana do melhor h",
            linewidth=1.8,
        )

        plt.plot(
            x_media,
            y_media,
            label="Mediana da média populacional",
            linewidth=1.8,
        )

        salvar_grafico(
            os.path.join(
                pasta_saida,
                f"melhor_vs_media_{instancia}.png",
            ),
            f"Melhor indivíduo versus média populacional — "
            f"{instancia} ({experimento})",
            "Número de avaliações da função objetivo",
            "Custo h",
        )

    # 5. Exporta os dados agregados para conferência.
    arquivo_agregado = os.path.join(
        pasta_saida, "series_agregadas.csv"
    )

    with open(
        arquivo_agregado,
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        escritor = csv.writer(arquivo)

        escritor.writerow([
            "instancia",
            "experimento",
            "avaliacoes",
            "mediana_melhor_h",
            "mediana_media_populacao",
            "mediana_diversidade",
        ])

        chaves = sorted({
            (
                linha["instancia"],
                linha["experimento"],
                linha["avaliacoes"],
            )
            for linha in dados
        })

        for instancia, experimento, avaliacoes in chaves:
            grupo = [
                linha for linha in dados
                if linha["instancia"] == instancia
                and linha["experimento"] == experimento
                and linha["avaliacoes"] == avaliacoes
            ]

            escritor.writerow([
                instancia,
                experimento,
                avaliacoes,
                median([linha["melhor_h"] for linha in grupo]),
                median([
                    linha["media_populacao"] for linha in grupo
                ]),
                median([
                    linha["diversidade"] for linha in grupo
                ]),
            ])

    print(f"Dados agregados salvos: {arquivo_agregado}")
    print("\nTodos os gráficos foram gerados.")
    print(
        "Observação: estes gráficos comparam as variantes do GA. "
        "Eles não substituem a comparação com Greedy e Simulated Annealing."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos dos resultados do algoritmo genético."
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="resultados_ga.csv",
        help="CSV de entrada (padrão: resultados_ga.csv)",
    )

    parser.add_argument(
        "--saida",
        default="graficos_ga",
        help="Pasta de saída (padrão: graficos_ga)",
    )

    args = parser.parse_args()

    try:
        dados = carregar_dados(args.csv)
        gerar_graficos(dados, args.saida)

    except (OSError, ValueError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
