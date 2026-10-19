# comparacao.py
import argparse
import csv
import os
import statistics
from bisect import bisect_right
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

ORC = {"pequena": 50_000, "media": 150_000, "grande": 300_000}
REF = {"pequena": 1875, "media": 16332, "grande": 60417}
H_OTIMO = 1875
PONTOS = 400
ORDEM = ["greedy", "sa", "ga"]
NOMES = {"greedy": "Greedy", "sa": "Simulated Annealing", "ga": "Genetic Algorithm"}


def ler(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def salvar_csv(caminho, linhas):
    if not linhas:
        return
    campos = []
    for l in linhas:
        for c in l:
            if c not in campos:
                campos.append(c)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(linhas)


def carregar(pasta):
    finais, hists, pops, rotulos = {}, {}, {}, {}

    def add_hist(alg, inst, chave, aval, h):
        hists.setdefault((alg, inst), {}).setdefault(chave, []).append((aval, h))

    # greedy
    for r in ler(os.path.join(pasta, "finais_greedy.csv")):
        finais.setdefault(("greedy", r["instancia"]), {})[int(r["seed"])] = float(r["melhor_h"])
        rotulos["greedy"] = f"Greedy ({r['vizinhanca']}/{r['variante']})"
    for r in ler(os.path.join(pasta, "historico_greedy.csv")):
        add_hist("greedy", r["instancia"], int(r["seed"]), int(float(r["avaliacoes"])), float(r["melhor_h"]))

    # simulated annealing (arquivos que o simulated_annealing.py precisa salvar)
    for r in ler(os.path.join(pasta, "finais_sa.csv")):
        finais.setdefault(("sa", r["instancia"]), {})[int(r["seed"])] = float(r["melhor_h"])
    for r in ler(os.path.join(pasta, "curvas_sa.csv")):
        add_hist("sa", r["instancia"], int(r["seed"]), int(float(r["avaliacoes"])), float(r["melhor_h"]))

    # genetic algorithm (só o experimento OX)
    for r in ler(os.path.join(pasta, "resultados_ga.csv")):
        if r["experimento"] != "OX":
            continue
        inst, rod = r["instancia"], int(r["rodada"])
        aval = int(float(r["avaliacoes"]))
        add_hist("ga", inst, rod, aval, float(r["melhor_h"]))
        pops.setdefault(inst, {}).setdefault(rod, []).append((aval, float(r["media_populacao"])))

    for d in list(hists.values()) + list(pops.values()):
        for k in d:
            d[k].sort()
    for (alg, inst), d in hists.items():       # GA: o resultado final é o último ponto
        if alg == "ga":
            finais[("ga", inst)] = {k: v[-1][1] for k, v in d.items()}
    return finais, hists, pops, rotulos


def curvas(historicos, orc, pontos=PONTOS):
    xs = [round(orc * k / pontos) for k in range(1, pontos + 1)]
    matriz = []
    for h in historicos:
        avals = [a for a, _ in h]
        matriz.append([h[max(bisect_right(avals, x) - 1, 0)][1] for x in xs])
    return xs, matriz


def mw(x, y):
    if len(set(x) | set(y)) == 1:
        return 1.0                              # valores idênticos: sem diferença
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def main(pasta):
    finais, hists, pops, rotulos = carregar(pasta)
    saida = os.path.join(pasta, "graficos_comparacao")
    os.makedirs(saida, exist_ok=True)
    rot = lambda a: rotulos.get(a, NOMES[a])
    lista = lambda a, i: [v for _, v in sorted(finais[(a, i)].items())]
    algs_de = lambda i: [a for a in ORDEM if (a, i) in finais]
    insts = [i for i in ORC if algs_de(i)]
    if not insts:
        print("nenhum resultado encontrado na pasta.")
        return

    def aval_ate(alg, inst):
        return [next((a for a, h in hist if h <= H_OTIMO), ORC[inst])
                for hist in hists.get((alg, inst), {}).values()]

    # 1) tabela principal
    tabela = []
    for inst in insts:
        for alg in algs_de(inst):
            v = lista(alg, inst)
            linha = {"instancia": inst, "algoritmo": rot(alg), "execucoes": len(v),
                     "media": round(statistics.mean(v), 1),
                     "desvio": round(statistics.stdev(v), 1) if len(v) > 1 else 0,
                     "min": min(v), "mediana": statistics.median(v)}
            if inst == "pequena":
                linha["gap_medio"] = round((statistics.mean(v) - H_OTIMO) / H_OTIMO, 4)
                linha["gap_min"] = round((min(v) - H_OTIMO) / H_OTIMO, 4)
                linha["execucoes_com_1875"] = sum(1 for x in v if x == H_OTIMO)
                av = aval_ate(alg, inst)
                if av:
                    linha["aval_mediana_ate_1875"] = statistics.median(av)
            tabela.append(linha)
            print(linha)
    salvar_csv(os.path.join(pasta, "tabela_principal.csv"), tabela)

    # 2) boxplots do melhor h final
    fig, axes = plt.subplots(1, len(insts), figsize=(5.2 * len(insts), 5), squeeze=False)
    for ax, inst in zip(axes[0], insts):
        algs = algs_de(inst)
        ax.boxplot([lista(a, inst) for a in algs])
        ax.set_xticks(range(1, len(algs) + 1))
        ax.set_xticklabels([rot(a) for a in algs], rotation=15)
        ax.set_title(f"Instância {inst}")
        ax.set_ylabel("melhor h")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(saida, "boxplots_melhor_h.png"), dpi=180)
    plt.close(fig)

    # 2b) pequena: todos acham 1875, então compara-se o custo para chegar lá
    if "pequena" in insts:
        algs = [a for a in algs_de("pequena") if aval_ate(a, "pequena")]
        if algs:
            plt.figure(figsize=(6, 5))
            plt.boxplot([aval_ate(a, "pequena") for a in algs])
            plt.xticks(range(1, len(algs) + 1), [rot(a) for a in algs], rotation=15)
            plt.ylabel("avaliações até h = 1875 (orçamento se não achou)")
            plt.title("Instância pequena: avaliações até o ótimo")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(saida, "boxplot_aval_ate_1875_pequena.png"), dpi=180)
            plt.close()

    # 3) curvas de convergência (mediana das execuções) + média da população do GA
    for inst in insts:
        plt.figure(figsize=(9.5, 5.8))
        for alg in algs_de(inst):
            if (alg, inst) not in hists:
                continue
            xs, matriz = curvas(list(hists[(alg, inst)].values()), ORC[inst])
            plt.plot(xs, [statistics.median(c) for c in zip(*matriz)], lw=2, label=rot(alg))
        if inst in pops:
            xs, matriz = curvas(list(pops[inst].values()), ORC[inst])
            plt.plot(xs, [statistics.median(c) for c in zip(*matriz)], lw=1.5, ls=":",
                     color="tab:green", label="GA: média da população")
        plt.axhline(REF[inst], color="tab:red", ls="--", lw=1, label=f"referência ({REF[inst]})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("avaliações de h")
        plt.ylabel("melhor h (mediana das execuções)")
        plt.title(f"Convergência: instância {inst}")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(saida, f"convergencia_{inst}.png"), dpi=180)
        plt.close()

    # 4) testes de Mann-Whitney entre algoritmos
    if mannwhitneyu is None:
        print("scipy não instalado (pip install scipy): testes pulados.")
        return
    testes = []
    for inst in insts:
        for a, b in combinations(algs_de(inst), 2):
            testes.append({"instancia": inst, "metrica": "melhor_h", "comparacao": f"{a} vs {b}",
                           "mediana_a": statistics.median(lista(a, inst)),
                           "mediana_b": statistics.median(lista(b, inst)),
                           "p_valor": float(f"{mw(lista(a, inst), lista(b, inst)):.3g}")})
            if inst == "pequena" and aval_ate(a, inst) and aval_ate(b, inst):
                x, y = aval_ate(a, inst), aval_ate(b, inst)
                testes.append({"instancia": inst, "metrica": "aval_ate_1875", "comparacao": f"{a} vs {b}",
                               "mediana_a": statistics.median(x), "mediana_b": statistics.median(y),
                               "p_valor": float(f"{mw(x, y):.3g}")})
    for t in testes:
        print(t)
    salvar_csv(os.path.join(pasta, "comparacao_testes.csv"), testes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", default=".")
    main(ap.parse_args().pasta)