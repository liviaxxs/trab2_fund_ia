import math
from incremental import h_inc

def greedy_inc(pb, orcamento, variante="best", guardar_otimos=False):
    if variante not in ("best", "first"):
        raise ValueError(f"variante invalida: {variante}")
    pb.resetar_contador()
    tipo = pb.vizinhanca

    melhor_global, melhor_s = math.inf, None
    otimos_locais, otimos_distintos = [], set()
    passos_por_restart, historico = [], []
    acabou = False

    def registrar(h_x, construir):
        nonlocal melhor_global, melhor_s
        if h_x < melhor_global:
            melhor_global = h_x
            melhor_s = construir()          # só monta a lista quando melhora
            historico.append((pb.n_avaliacoes, h_x))

    while not acabou and not pb.esgotado(orcamento):
        s = pb.solucao_aleatoria()
        cache = pb.criar_cache(s)           # conta 1 avaliação
        h_s = pb.custo_do_cache(cache)
        registrar(h_s, lambda: s)
        passos = 0

        while True:
            melhor_mov, h_melhor = None, h_s
            for i, j in pb._movimentos(tipo):
                if pb.esgotado(orcamento):
                    acabou = True
                    break
                h_t = h_inc(pb, s, cache, tipo, i, j)
                registrar(h_t, lambda: pb.aplicar(s, tipo, i, j))
                if h_t < h_melhor:
                    melhor_mov, h_melhor = (i, j), h_t
                    if variante == "first":
                        break

            if acabou:
                passos_por_restart.append(passos)
                break
            if melhor_mov is None:          # varredura completa sem melhora
                otimos_locais.append(h_s)
                if guardar_otimos:
                    otimos_distintos.add(tuple(s))
                passos_por_restart.append(passos)
                break

            i, j = melhor_mov
            s = pb.aplicar(s, tipo, i, j)
            cache = pb.atualizar_cache(s, cache, pb.posicao_inicial(tipo, i, j))
            h_s = h_melhor
            passos += 1

    return {"melhor_h": melhor_global, "melhor_s": melhor_s,
            "otimos_locais": otimos_locais,
            "otimos_distintos": otimos_distintos,
            "passos_por_restart": passos_por_restart,
            "historico": historico, "n_avaliacoes": pb.n_avaliacoes}