"""
tp2 fundamentos de ia - simulated annealing.
"""
import math
import statistics
import time
from problema import ProblemaManutencao


def amostrar_deltas(pb, n_solucoes=200, movs_por_solucao=50, vizinhanca="swap"):
    """amostra pioras (delta > 0) em solucoes aleatorias pra estimar escala."""
    pos = []
    for _ in range(n_solucoes):
        s = pb.solucao_aleatoria()
        hs = pb._custo(s)
        for _ in range(movs_por_solucao):
            i, j = pb.movimento_aleatorio(vizinhanca)
            d = pb._custo(pb.aplicar(s, vizinhanca, i, j)) - hs
            if d > 0:
                pos.append(d)
    return pos


def calibrar_schedule(pb, orcamento, p0=0.8, Tf=2.0):
    """t0 aceita a piora media com prob p0; alpha leva t0 ate tf em `orcamento` passos."""
    pos = amostrar_deltas(pb)
    media = statistics.mean(pos)
    # exp(-media/t0) = p0
    T0 = -media / math.log(p0)
    alpha = (Tf / T0) ** (1.0 / orcamento)
    return dict(T0=T0, Tf=Tf, alpha=alpha, delta_medio=media, p0=p0)


def simulated_annealing(pb, orcamento, T0, alpha, janela=None, vizinhanca="swap"):
    """uma rodada de sa com orcamento fixo de avaliacoes."""
    janela = janela or max(1, orcamento // 100)
    pb.resetar_contador()

    s = pb.solucao_aleatoria()
    cache = pb.criar_cache(s)
    h_atual = pb.custo_do_cache(cache)
    melhor_s, melhor_h = s[:], h_atual

    T = T0

    curva = [(pb.n_avaliacoes, melhor_h)]
    aceit_piora = []
    pioras_jan = aceitas_jan = 0
    aceit_geral_jan = 0
    passos_jan = 0

    while not pb.esgotado(orcamento):
        # avalia o vizinho de forma incremental, sem refazer o custo inteiro
        i, j = pb.movimento_aleatorio(vizinhanca)
        h_novo = pb.h_swap(s, cache, i, j)
        delta = h_novo - h_atual

        aceitou = False
        if delta <= 0:
            aceitou = True
        else:
            pioras_jan += 1
            if pb.rng.random() < math.exp(-delta / T):
                aceitou = True
                aceitas_jan += 1

        if aceitou:
            s_novo = pb.aplicar(s, vizinhanca, i, j)
            cache = pb.atualizar_cache(s_novo, cache, pb.posicao_inicial(vizinhanca, i, j))
            s, h_atual = s_novo, h_novo
            aceit_geral_jan += 1
            if h_atual < melhor_h:
                melhor_s, melhor_h = s[:], h_atual
                curva.append((pb.n_avaliacoes, melhor_h))

        passos_jan += 1
        T *= alpha

        if passos_jan == janela:
            taxa = aceitas_jan / pioras_jan if pioras_jan else float("nan")
            aceit_piora.append((pb.n_avaliacoes, T, taxa, aceit_geral_jan / passos_jan))
            pioras_jan = aceitas_jan = aceit_geral_jan = passos_jan = 0

    curva.append((pb.n_avaliacoes, melhor_h))
    return dict(melhor_s=melhor_s, melhor_h=melhor_h, curva=curva,
                aceitacao=aceit_piora, T_final=T, avaliacoes=pb.n_avaliacoes)


CONFIG = {"pequena": (50_000, 30), "media": (150_000, 30), "grande": (300_000, 10)}
H_OTIMO_PEQUENA = 1875
SEED_CALIB, SEED_BASE = 12345, 1000


def main(pasta="."):
    import os

    for nome, (orc, nrun) in CONFIG.items():
        arq = os.path.join(pasta, f"manutencao_{nome}.txt")
        if not os.path.exists(arq):
            print(f"arquivo nao encontrado: {arq}")
            continue

        pb0 = ProblemaManutencao(arq, seed=SEED_CALIB)
        if nome == "pequena":
            # h(identidade) deve dar 4366
            assert pb0._custo(list(range(1, pb0.n + 1))) == 4366, \
                "h(identidade) deveria ser 4366"
        cfg = calibrar_schedule(pb0, orc)
        print(f"[{nome}] t0={cfg['T0']:.1f} tf={cfg['Tf']} alpha={cfg['alpha']:.8f}")

        hs, tempos = [], []
        for r in range(nrun):
            pb = ProblemaManutencao(arq, seed=SEED_BASE + r)
            t0 = time.perf_counter()
            out = simulated_annealing(pb, orc, cfg["T0"], cfg["alpha"])
            tempos.append(time.perf_counter() - t0)

            assert out["avaliacoes"] == orc
            assert sorted(out["melhor_s"]) == list(range(1, pb.n + 1))
            assert pb._custo(out["melhor_s"]) == out["melhor_h"]

            hs.append(out["melhor_h"])

        # stdev amostral, porque n e pequeno
        res = dict(
            instancia=nome,
            media=round(statistics.mean(hs), 1),
            desvio=round(statistics.stdev(hs), 1),
            minimo=min(hs),
            mediana=statistics.median(hs),
            tempo_medio_s=round(statistics.mean(tempos), 2),
        )
        print("  ", res)

        if nome == "pequena":
            ach = sum(1 for v in hs if v == H_OTIMO_PEQUENA)
            gap = (statistics.mean(hs) - H_OTIMO_PEQUENA) / H_OTIMO_PEQUENA
            print(f"   gap medio = {gap:.4f} | execucoes que acharam {H_OTIMO_PEQUENA}: {ach}/{nrun}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
