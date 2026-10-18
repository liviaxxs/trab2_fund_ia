import math
import problema


def greedy_descent(p, orcamento, guardar_otimos=False):
    p.resetar_contador()

    melhor_global = math.inf
    melhor_s = None
    otimos_locais = []          # h de cada ótimo local (descidas completas)
    otimos_distintos = set()    # tuple(s) de cada ótimo local
    passos_por_restart = []     # nº de passos de melhoria por restart
    historico = []              # (avaliações gastas, melhor h) a cada melhora
    acabou = False

    def registrar(h_x, x):
        """Atualiza o melhor global sempre que uma solução avaliada o supera."""
        nonlocal melhor_global, melhor_s
        if h_x < melhor_global:
            melhor_global = h_x
            melhor_s = x
            historico.append((p.n_avaliacoes, h_x))

    while not acabou and not p.esgotado(orcamento):
        s = p.solucao_aleatoria()
        h_s = p.h(s)
        registrar(h_s, s)
        passos = 0

        while True:
            melhor_vizinho = None
            h_melhor = h_s

            for sucessor in p.sucessores(s):
                if p.esgotado(orcamento):
                    acabou = True
                    break
                h_t = p.h(sucessor)
                registrar(h_t, sucessor)       # todo sucessor avaliado é "visto"
                if h_t < h_melhor:
                    melhor_vizinho = sucessor
                    h_melhor = h_t

            if acabou:
                passos_por_restart.append(passos)   # restart interrompido (parcial)
                break

            if melhor_vizinho is None:              # varredura completa, sem melhora
                otimos_locais.append(h_s)
                if guardar_otimos:
                    otimos_distintos.add(tuple(s))
                passos_por_restart.append(passos)
                break

            s = melhor_vizinho
            h_s = h_melhor
            passos += 1

    return {
        "melhor_h": melhor_global,
        "melhor_s": melhor_s,
        "otimos_locais": otimos_locais,
        "n_otimos_distintos": len(otimos_distintos),
        "passos_por_restart": passos_por_restart,
        "historico": historico,
        "n_avaliacoes": p.n_avaliacoes,
    }