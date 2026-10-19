# t3_greedy.py
import argparse
import csv
import os
import statistics
import time
from bisect import bisect_right

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problema import ProblemaManutencao
from greedy_descent_random import greedy_inc

INSTANCIAS = [("pequena", 50_000, 30), ("media", 150_000, 30), ("grande", 300_000, 10)]
REFERENCIA = {"pequena": 1875, "media": 16332, "grande": 60417}
H_OTIMO = 1875
PONTOS = 400


def salvar_csv(caminho, linhas):
    if not linhas:
        return
    campos = []
    for l in linhas:                          # união das colunas (a pequena tem extras)
        for c in l:
            if c not in campos:
                campos.append(c)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(linhas)


def curvas(historicos, orc, pontos=PONTOS):
    """Melhor h em função das avaliações, amostrado numa grade comum (função em degraus)."""
    xs = [round(orc * k / pontos) for k in range(1, pontos + 1)]
    matriz = []
    for h in historicos:
        avals = [a for a, _ in h]
        linha = []
        for x in xs:
            idx = max(bisect_right(avals, x) - 1, 0)
            linha.append(h[idx][1])
        matriz.append(linha)
    return xs, matriz


def rodar(pasta, viz, var, rapido):
    saida = os.path.join(pasta, "graficos_greedy")
    os.makedirs(saida, exist_ok=True)
    finais, brutos, resumos, curvas_med = [], [], [], []

    for nome, orc, nexec in INSTANCIAS:
        caminho = os.path.join(pasta, f"manutencao_{nome}.txt")
        if not os.path.exists(caminho):
            print(f"arquivo não encontrado: {caminho}")
            continue
        if rapido:
            nexec = 3

        valores, tempos, restarts, passos, historicos = [], [], [], [], []
        avals, n_otimos, distintos = [], 0, set()

        for seed in range(nexec):
            pb = ProblemaManutencao(caminho, vizinhanca=viz, seed=seed)
            t0 = time.perf_counter()
            r = greedy_inc(pb, orc, var, guardar_otimos=(nome == "pequena"))
            tempo = time.perf_counter() - t0

            assert r["n_avaliacoes"] == orc, "orçamento não foi respeitado"
            assert pb._custo(r["melhor_s"]) == r["melhor_h"], "melhor_s não bate com melhor_h"

            aval = ""
            if nome == "pequena":
                aval = next((a for a, h in r["historico"] if h == H_OTIMO), orc)
                avals.append(aval)
                distintos |= r["otimos_distintos"]

            valores.append(r["melhor_h"])
            tempos.append(tempo)
            restarts.append(len(r["passos_por_restart"]))
            passos.append(sum(r["passos_por_restart"]))
            n_otimos += len(r["otimos_locais"])
            historicos.append(r["historico"])

            finais.append({"algoritmo": "greedy", "vizinhanca": viz, "variante": var,
                           "instancia": nome, "seed": seed, "melhor_h": r["melhor_h"],
                           "tempo_s": round(tempo, 3), "avaliacoes": orc,
                           "aval_ate_1875": aval})
            for a, h in r["historico"]:
                brutos.append({"instancia": nome, "seed": seed, "avaliacoes": a, "melhor_h": h})

        linha = {
            "algoritmo": "greedy", "vizinhanca": viz, "variante": var, "instancia": nome,
            "execucoes": nexec, "orcamento": orc,
            "media": round(statistics.mean(valores), 1),
            "desvio": round(statistics.stdev(valores), 1),
            "min": min(valores), "mediana": statistics.median(valores),
            "tempo_medio_s": round(statistics.mean(tempos), 2),
            "restarts_medios": round(statistics.mean(restarts), 1),
            "passos_medios": round(statistics.mean(passos), 1),
            "n_otimos_locais": n_otimos,
        }
        if nome == "pequena":
            linha["gap_medio"] = round((statistics.mean(valores) - H_OTIMO) / H_OTIMO, 4)
            linha["gap_min"] = round((min(valores) - H_OTIMO) / H_OTIMO, 4)
            linha["otimos_distintos"] = len(distintos)
            linha["execucoes_com_1875"] = sum(1 for v in valores if v == H_OTIMO)
            linha["aval_media_ate_1875"] = round(statistics.mean(avals))
        resumos.append(linha)
        print(linha, flush=True)

        # curva de convergência (mediana das execuções, uma linha fraca por execução)
        xs, matriz = curvas(historicos, orc)
        med = [statistics.median(col) for col in zip(*matriz)]
        for x, m in zip(xs, med):
            curvas_med.append({"instancia": nome, "avaliacoes": x, "mediana_melhor_h": m})

        plt.figure(figsize=(9, 5.5))
        for linha_run in matriz:
            plt.plot(xs, linha_run, color="tab:blue", alpha=0.12, lw=0.8)
        plt.plot(xs, med, color="tab:blue", lw=2, label=f"mediana ({nexec} execuções)")
        plt.axhline(REFERENCIA[nome], color="tab:red", ls="--", lw=1,
                    label=f"referência ({REFERENCIA[nome]})")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("avaliações de h")
        plt.ylabel("melhor h encontrado")
        plt.title(f"Greedy ({viz}/{var}): convergência, instância {nome}")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(saida, f"convergencia_{nome}.png"), dpi=180)
        plt.close()

    salvar_csv(os.path.join(pasta, "finais_greedy.csv"), finais)
    salvar_csv(os.path.join(pasta, "historico_greedy.csv"), brutos)
    salvar_csv(os.path.join(pasta, "curvas_greedy.csv"), curvas_med)
    salvar_csv(os.path.join(pasta, "resumo_greedy.csv"), resumos)
    print("\nGerados: finais_greedy.csv, historico_greedy.csv, curvas_greedy.csv, "
            "resumo_greedy.csv e graficos_greedy/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", nargs="?", default=".")
    ap.add_argument("--viz", choices=ProblemaManutencao.VIZINHANCAS, default="2opt")
    ap.add_argument("--var", choices=("best", "first"), default="best")
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()
    rodar(a.pasta, a.viz, a.var, a.rapido)