# t2.py
import csv
import os
import statistics
import sys
import time
from itertools import combinations

from problema import ProblemaManutencao
from greedy_descent_random import greedy_inc

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None

INSTANCIAS = [("pequena", 50_000, 30), ("media", 150_000, 30), ("grande", 300_000, 10)]
VARIANTES = ("best", "first")
VIZ = ProblemaManutencao.VIZINHANCAS
H_OTIMO = 1875


def salvar_csv(caminho, linhas):
    if not linhas:
        return
    campos = []
    for l in linhas:                       # união das colunas (a pequena tem colunas extras)
        for c in l:
            if c not in campos:
                campos.append(c)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, restval="")
        w.writeheader()
        w.writerows(linhas)


def mw(x, y):
    """Mann-Whitney bilateral. Valores todos idênticos: sem diferença (p = 1)."""
    if len(set(x) | set(y)) == 1:
        return 1.0
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def comparar(finais, metrica):
    testes = []
    for nome in sorted({k[0] for k in finais}):
        for var in VARIANTES:                                  # entre vizinhanças
            for a, b in combinations(VIZ, 2):
                p = mw(finais[(nome, var, a)], finais[(nome, var, b)])
                testes.append({"metrica": metrica, "instancia": nome,
                               "comparacao": f"{var}: {a} vs {b}", "p_valor": round(p, 6)})
        for tipo in VIZ:                                       # entre variantes
            p = mw(finais[(nome, "best", tipo)], finais[(nome, "first", tipo)])
            testes.append({"metrica": metrica, "instancia": nome,
                           "comparacao": f"{tipo}: best vs first", "p_valor": round(p, 6)})
    return testes


def rodar(pasta, rapido=False):
    execucoes, resumos = [], []
    finais, finais_aval = {}, {}

    for nome, orc, nexec in INSTANCIAS:
        caminho = os.path.join(pasta, f"manutencao_{nome}.txt")
        if not os.path.exists(caminho):
            print(f"arquivo não encontrado: {caminho}")
            continue
        if rapido:
            nexec = 3

        for var in VARIANTES:
            for tipo in VIZ:
                valores, tempos, passos, avals = [], [], [], []
                otimos_todos, distintos = [], set()

                for seed in range(nexec):          # mesmas seeds em todas as configurações
                    pb = ProblemaManutencao(caminho, vizinhanca=tipo, seed=seed)
                    t0 = time.perf_counter()
                    r = greedy_inc(pb, orc, var, guardar_otimos=(nome == "pequena"))
                    tempo = time.perf_counter() - t0

                    assert r["n_avaliacoes"] == orc, "orçamento não foi respeitado"
                    assert pb._custo(r["melhor_s"]) == r["melhor_h"], "melhor_s não bate com melhor_h"

                    aval = ""
                    if nome == "pequena":
                        # avaliações até o 1875; se não achou, conta o orçamento inteiro
                        aval = next((a for a, h in r["historico"] if h == H_OTIMO), orc)
                        avals.append(aval)

                    valores.append(r["melhor_h"])
                    tempos.append(tempo)
                    passos.append(sum(r["passos_por_restart"]))
                    otimos_todos += r["otimos_locais"]
                    distintos |= r["otimos_distintos"]

                    execucoes.append({
                        "instancia": nome, "variante": var, "vizinhanca": tipo,
                        "seed": seed, "melhor_h": r["melhor_h"],
                        "aval_ate_1875": aval,
                        "n_restarts": len(r["passos_por_restart"]),
                        "passos_total": passos[-1],
                        "n_otimos_locais": len(r["otimos_locais"]),
                        "tempo_s": round(tempo, 3),
                    })

                finais[(nome, var, tipo)] = valores
                linha = {
                    "instancia": nome, "variante": var, "vizinhanca": tipo,
                    "tam_vizinhanca": pb.tamanho_vizinhanca(tipo),
                    "execucoes": nexec,
                    "media": round(statistics.mean(valores), 1),
                    "desvio": round(statistics.stdev(valores), 1) if nexec > 1 else 0,
                    "min": min(valores),
                    "mediana": statistics.median(valores),
                    "n_otimos_locais": len(otimos_todos),
                    "media_otimos_locais": round(statistics.mean(otimos_todos), 1) if otimos_todos else "",
                    "passos_medios": round(statistics.mean(passos), 1),
                    "tempo_medio_s": round(statistics.mean(tempos), 2),
                    "us_por_avaliacao": round(statistics.mean(tempos) / orc * 1e6, 2),
                }
                if nome == "pequena":
                    finais_aval[(nome, var, tipo)] = avals
                    linha["otimos_distintos"] = len(distintos)
                    linha["execucoes_com_1875"] = sum(1 for v in valores if v == H_OTIMO)
                    linha["aval_media_ate_1875"] = round(statistics.mean(avals))
                    linha["aval_mediana_ate_1875"] = statistics.median(avals)
                resumos.append(linha)
                print(linha, flush=True)

    salvar_csv(os.path.join(pasta, "resultados_t2_execucoes.csv"), execucoes)
    salvar_csv(os.path.join(pasta, "resultados_t2.csv"), resumos)

    if mannwhitneyu is None:
        print("\nscipy não instalado (pip install scipy): testes estatísticos pulados.")
        return
    testes = comparar(finais, "melhor_h") + comparar(finais_aval, "aval_ate_1875")
    print("\nMann-Whitney (bilateral):")
    for t in testes:
        print(f"{t['metrica']:14s} {t['instancia']:8s} {t['comparacao']:28s} p = {t['p_valor']}")
    salvar_csv(os.path.join(pasta, "resultados_t2_testes.csv"), testes)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pasta = args[0] if args else "."
    rodar(pasta, rapido="--rapido" in sys.argv)