
"""
TP2 - Fundamentos de IA
Algoritmo Genetico para sequenciamento de manutencao.

Dependencia: problema.py
Saidas:
    resultados_ga.csv
    resumo_ga.csv
"""

import csv
import random
import statistics
import sys
import time
from pathlib import Path

from problema import ProblemaManutencao


# Configuracoes experimentais
POPULACAO = 100
PC = 0.90
PM = 0.15
TAMANHO_TORNEIO = 3

ORCAMENTOS = {
    "pequena": 50_000,
    "media": 150_000,
    "grande": 300_000,
}

REPETICOES = {
    "pequena": 30,
    "media": 30,
    "grande": 10,
}


def ox(p1, p2, rng):
    """Order Crossover: retorna dois filhos validos."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))

    def filho(base, doador):
        f = [None] * n
        f[a:b + 1] = base[a:b + 1]
        usados = set(f[a:b + 1])

        # Percorre o doador circularmente apos o corte.
        genes = doador[b + 1:] + doador[:b + 1]
        posicoes = list(range(b + 1, n)) + list(range(0, a))

        restantes = [gene for gene in genes if gene not in usados]

        for pos, gene in zip(posicoes, restantes):
            f[pos] = gene

        return f

    return filho(p1, p2), filho(p2, p1)


def mutacao_swap(individuo, rng):
    """Troca duas posicoes e devolve uma nova permutacao."""
    filho = individuo.copy()
    i, j = rng.sample(range(len(filho)), 2)
    filho[i], filho[j] = filho[j], filho[i]
    return filho


def selecao_torneio(populacao, custos, rng):
    """Escolhe o melhor entre tres individuos sorteados."""
    candidatos = rng.sample(range(len(populacao)), TAMANHO_TORNEIO)
    vencedor = min(candidatos, key=lambda i: custos[i])
    return populacao[vencedor]


def executar_ga(
    pb,
    orcamento,
    seed,
    pc=PC,
    pm=PM,
    tamanho_populacao=POPULACAO,
):
    """
    Executa uma rodada do GA.

    O custo de cada individuo e contabilizado por pb.h().
    O orcamento deve ser divisivel pelo tamanho da populacao.
    """
    if orcamento % tamanho_populacao != 0:
        raise ValueError("Orcamento deve ser divisivel pela populacao.")

    rng = random.Random(seed)
    n = pb.n

    # Inicializacao: cada individuo e uma permutacao.
    populacao = []
    for _ in range(tamanho_populacao):
        individuo = list(range(1, n + 1))
        rng.shuffle(individuo)
        populacao.append(individuo)

    inicio_tempo = time.perf_counter()
    pb.resetar_contador()

    custos = [pb.h(ind) for ind in populacao]

    melhor_idx = min(range(len(custos)), key=lambda i: custos[i])
    melhor_solucao = populacao[melhor_idx].copy()
    melhor_custo = custos[melhor_idx]

    historico = []

    def registrar():
        historico.append({
            "avaliacoes": pb.n_avaliacoes,
            "melhor_h": melhor_custo,
            "media_populacao": statistics.mean(custos),
            "diversidade": len({tuple(ind) for ind in populacao}),
        })

    registrar()

    while pb.n_avaliacoes < orcamento:
        # Elitismo: preserva o melhor individuo da geracao.
        elite_idx = min(range(len(custos)), key=lambda i: custos[i])
        nova_populacao = [populacao[elite_idx].copy()]

        # Produz filhos ate completar a populacao.
        while len(nova_populacao) < tamanho_populacao:
            p1 = selecao_torneio(populacao, custos, rng)
            p2 = selecao_torneio(populacao, custos, rng)

            if rng.random() < pc:
                f1, f2 = ox(p1, p2, rng)
            else:
                f1, f2 = p1.copy(), p2.copy()

            for filho in (f1, f2):
                if rng.random() < pm:
                    filho = mutacao_swap(filho, rng)

                nova_populacao.append(filho)

                if len(nova_populacao) == tamanho_populacao:
                    break

        populacao = nova_populacao

        # Avalia todos os individuos da nova geracao.
        custos = [pb.h(ind) for ind in populacao]

        idx = min(range(len(custos)), key=lambda i: custos[i])
        if custos[idx] < melhor_custo:
            melhor_custo = custos[idx]
            melhor_solucao = populacao[idx].copy()

        registrar()

    tempo = time.perf_counter() - inicio_tempo

    return {
        "melhor_h": melhor_custo,
        "melhor_solucao": melhor_solucao,
        "avaliacoes": pb.n_avaliacoes,
        "tempo_s": tempo,
        "historico": historico,
    }


def salvar_csv(caminho, linhas):
    if not linhas:
        return

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo, fieldnames=list(linhas[0].keys())
        )
        escritor.writeheader()
        escritor.writerows(linhas)


def executar_experimentos(pasta="."):
    pasta = Path(pasta)
    resultados = []
    resumos = []

    for nome in ("pequena", "media", "grande"):
        arquivo = pasta / f"manutencao_{nome}.txt"

        if not arquivo.exists():
            print(f"Arquivo nao encontrado: {arquivo}")
            continue

        orcamento = ORCAMENTOS[nome]
        repeticoes = REPETICOES[nome]

        # Pc=0 e experimento de controle sem recombinacao.
        configuracoes = [
            ("OX", PC),
            ("sem_cruzamento", 0.0),
        ]

        for experimento, pc in configuracoes:
            melhores = []

            for rodada in range(repeticoes):
                seed = 10_000 + rodada
                pb = ProblemaManutencao(
                    str(arquivo), seed=seed
                )

                resultado = executar_ga(
                    pb,
                    orcamento=orcamento,
                    seed=seed,
                    pc=pc,
                )

                melhores.append(resultado["melhor_h"])

                for ponto in resultado["historico"]:
                    resultados.append({
                        "instancia": nome,
                        "experimento": experimento,
                        "rodada": rodada + 1,
                        "seed": seed,
                        **ponto,
                    })

                print(
                    f"{nome:8s} | {experimento:16s} | "
                    f"rodada {rodada + 1:2d}/{repeticoes} | "
                    f"melhor h = {resultado['melhor_h']} | "
                    f"avaliacoes = {resultado['avaliacoes']} | "
                    f"tempo = {resultado['tempo_s']:.2f} s"
                )

            resumos.append({
                "instancia": nome,
                "experimento": experimento,
                "n_execucoes": len(melhores),
                "media": statistics.mean(melhores),
                "desvio_padrao": (
                    statistics.stdev(melhores)
                    if len(melhores) > 1 else 0.0
                ),
                "minimo": min(melhores),
                "mediana": statistics.median(melhores),
            })

    salvar_csv(pasta / "resultados_ga.csv", resultados)
    salvar_csv(pasta / "resumo_ga.csv", resumos)

    print("\nArquivos gerados:")
    print(pasta / "resultados_ga.csv")
    print(pasta / "resumo_ga.csv")


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    executar_experimentos(pasta)
