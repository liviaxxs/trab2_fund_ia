import math
import os
import sys

from problema import ProblemaManutencao
from incremental import h_inc
from greedy_descent_random import greedy_inc


def testar_h_inc(caminho):
    """h_inc == avaliação completa, nas 3 vizinhanças."""
    pb = ProblemaManutencao(caminho, seed=0)
    total = 0
    for tipo in ProblemaManutencao.VIZINHANCAS:
        movs = list(pb._movimentos(tipo))
        if len(movs) > 3000:
            movs = pb.rng.sample(movs, 3000)
        for _ in range(5):                      # 5 soluções diferentes
            s = pb.solucao_aleatoria()
            cache = pb.criar_cache(s)
            for i, j in movs:
                antes = pb.n_avaliacoes
                inc = h_inc(pb, s, cache, tipo, i, j)
                assert pb.n_avaliacoes == antes + 1, "h_inc deve contar 1 avaliação"
                completo = pb._custo(pb.aplicar(s, tipo, i, j))
                assert inc == completo, f"{tipo} ({i},{j}): inc {inc} != completo {completo}"
                total += 1
    print(f"  OK: h_inc == completa em {total} movimentos (swap, insertion, 2opt)")


def greedy_referencia(pb, orcamento, variante):
    """Mesmo algoritmo, mas com pb.h e pb.sucessores (avaliação completa)."""
    pb.resetar_contador()
    melhor = math.inf
    passos_por_restart, otimos = [], []
    acabou = False
    while not acabou and not pb.esgotado(orcamento):
        s = pb.solucao_aleatoria()
        h_s = pb.h(s)
        melhor = min(melhor, h_s)
        passos = 0
        while True:
            melhor_viz, h_melhor = None, h_s
            for t in pb.sucessores(s):
                if pb.esgotado(orcamento):
                    acabou = True
                    break
                h_t = pb.h(t)
                melhor = min(melhor, h_t)
                if h_t < h_melhor:
                    melhor_viz, h_melhor = t, h_t
                    if variante == "first":
                        break
            if acabou:
                passos_por_restart.append(passos)
                break
            if melhor_viz is None:
                otimos.append(h_s)
                passos_por_restart.append(passos)
                break
            s, h_s = melhor_viz, h_melhor
            passos += 1
    return {"melhor_h": melhor, "passos_por_restart": passos_por_restart,
            "otimos_locais": otimos, "n_avaliacoes": pb.n_avaliacoes}


def testar_greedy(caminho, orcamento=20_000):
    """greedy_inc deve seguir a MESMA trajetória da versão com avaliação completa."""
    for tipo in ProblemaManutencao.VIZINHANCAS:
        for var in ("best", "first"):
            a = greedy_referencia(ProblemaManutencao(caminho, tipo, seed=3), orcamento, var)
            r = greedy_inc(ProblemaManutencao(caminho, tipo, seed=3), orcamento, var)
            assert r["n_avaliacoes"] == orcamento, f"{tipo}/{var}: orçamento {r['n_avaliacoes']}"
            assert r["melhor_h"] == a["melhor_h"], f"{tipo}/{var}: melhor_h difere"
            assert r["passos_por_restart"] == a["passos_por_restart"], f"{tipo}/{var}: passos diferem"
            assert r["otimos_locais"] == a["otimos_locais"], f"{tipo}/{var}: ótimos locais diferem"
            if "pequena" in caminho:
                assert r["melhor_h"] >= 1875, f"{tipo}/{var}: abaixo do ótimo global!"
    print(f"  OK: greedy_inc == referência completa (3 vizinhanças x best/first, orçamento {orcamento})")


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    for nome in ("pequena", "media", "grande"):
        caminho = os.path.join(pasta, f"manutencao_{nome}.txt")
        if not os.path.exists(caminho):
            continue
        print(f"=== {nome} ===")
        testar_h_inc(caminho)
        if nome != "grande":                    # na grande a referência completa é lenta
            testar_greedy(caminho)
    print("\nTudo certo.")