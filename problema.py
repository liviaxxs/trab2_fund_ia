"""
TP2 Fundamentos de IA: nucleo do problema de sequenciamento de manutencao.

Parte 1: leitura, h (com contador unico), 3 vizinhancas e interface comum.
Parte 2: avaliacao incremental (swap), teste automatico e medicao de tempo.

Uso: python problema.py [pasta_com_as_instancias]
Gera tambem resultados_nucleo.csv (tempos e tamanhos das vizinhancas).

Contagem de avaliacoes: toda chamada que devolve o custo de uma solucao
(h, criar_cache, h_swap) passa por _contar(), o unico ponto que incrementa
n_avaliacoes. atualizar_cache NAO conta (o custo ja foi contado ao avaliar o
sucessor aceito).
"""
import csv
import math
import os
import random
import sys
import time


def _num(v):
    """Devolve int quando o valor for inteiro, senao float."""
    return int(v) if v == int(v) else v


class ProblemaManutencao:
    VIZINHANCAS = ("swap", "insertion", "2opt")

    def __init__(self, caminho, vizinhanca="swap", seed=None):
        if vizinhanca not in self.VIZINHANCAS:
            raise ValueError(f"vizinhanca invalida: {vizinhanca}")
        self.vizinhanca = vizinhanca
        self.rng = random.Random(seed)
        self.n_avaliacoes = 0
        self._ler_instancia(caminho)
        self._calcular_deslocamentos()

    # ------------------------------------------------------------------
    # Leitura e pre-calculo
    # ------------------------------------------------------------------
    def _ler_instancia(self, caminho):
        with open(caminho) as f:
            tokens = f.read().split()
        self.n = int(tokens[0])
        valores = [float(t) for t in tokens[1:]]
        if len(valores) != 2 + 5 * self.n:
            raise ValueError(
                f"{caminho}: esperado {2 + 5 * self.n} valores apos n, "
                f"encontrado {len(valores)}"
            )
        # indice 0 = base; ativos nos indices 1..n
        self.x = [valores[0]]
        self.y = [valores[1]]
        self.p = [0]
        self.d = [0]
        self.w = [0]
        for j in range(self.n):
            xj, yj, pj, dj, wj = valores[2 + 5 * j: 7 + 5 * j]
            self.x.append(xj)
            self.y.append(yj)
            self.p.append(_num(pj))
            self.d.append(_num(dj))
            self.w.append(_num(wj))

    def _calcular_deslocamentos(self):
        """t[i][j] = round(distancia euclidiana), arredondando para o mais proximo."""
        m = self.n + 1
        self.t = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                dist = math.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j])
                self.t[i][j] = int(math.floor(dist + 0.5))

    # ------------------------------------------------------------------
    # Avaliacao completa e contador (unico lugar onde se conta)
    # ------------------------------------------------------------------
    def _contar(self):
        self.n_avaliacoes += 1

    def _custo(self, s):
        """Atraso ponderado total, sem mexer no contador."""
        tempo = 0
        atual = 0
        custo = 0
        for a in s:
            tempo += self.t[atual][a] + self.p[a]
            atraso = tempo - self.d[a]
            if atraso > 0:
                custo += self.w[a] * atraso
            atual = a
        return custo

    def h(self, s):
        """Fitness (minimizacao). Cada chamada conta 1 avaliacao."""
        self._contar()
        return self._custo(s)

    def resetar_contador(self):
        self.n_avaliacoes = 0

    def esgotado(self, orcamento):
        return self.n_avaliacoes >= orcamento

    # ------------------------------------------------------------------
    # Interface comum
    # ------------------------------------------------------------------
    def solucao_aleatoria(self):
        s = list(range(1, self.n + 1))
        self.rng.shuffle(s)
        return s

    def sucessores(self, s, vizinhanca=None):
        """Itera sobre todos os sucessores de s (listas novas)."""
        tipo = vizinhanca or self.vizinhanca
        for i, j in self._movimentos(tipo):
            yield self._aplicar(s, tipo, i, j)

    def sucessor_aleatorio(self, s, vizinhanca=None):
        """Sorteia um unico sucessor de s."""
        tipo = vizinhanca or self.vizinhanca
        i, j = self._movimento_aleatorio(tipo)
        return self._aplicar(s, tipo, i, j)

    def cruzamento(self, s1, s2):
        """Preenchido pela pessoa do GA (OX ou PMX)."""
        raise NotImplementedError("cruzamento: implementar no GA")

    def mutacao(self, s):
        """Swap mutation: troca duas posicoes sorteadas."""
        i, j = self._movimento_aleatorio("swap")
        return self._aplicar(s, "swap", i, j)

    # ------------------------------------------------------------------
    # Vizinhancas
    # ------------------------------------------------------------------
    def tamanho_vizinhanca(self, vizinhanca=None):
        tipo = vizinhanca or self.vizinhanca
        n = self.n
        if tipo == "insertion":
            return (n - 1) ** 2
        return n * (n - 1) // 2  # swap e 2opt

    def _movimentos(self, tipo):
        """Gera os pares (i, j) de posicoes (indices 0..n-1) de cada movimento."""
        n = self.n
        if tipo in ("swap", "2opt"):
            for i in range(n - 1):
                for j in range(i + 1, n):
                    yield i, j
        else:  # insertion: tira o elemento da posicao i e insere na posicao j
            for i in range(n):
                for j in range(n):
                    # j == i e nulo; j == i+1 repete o swap adjacente (j == i-1)
                    if j != i and j != i + 1:
                        yield i, j

    def _movimento_aleatorio(self, tipo):
        n = self.n
        if tipo in ("swap", "2opt"):
            i, j = sorted(self.rng.sample(range(n), 2))
            return i, j
        while True:
            i = self.rng.randrange(n)
            j = self.rng.randrange(n)
            if j != i and j != i + 1:
                return i, j

    @staticmethod
    def _aplicar(s, tipo, i, j):
        novo = s[:]
        if tipo == "swap":
            novo[i], novo[j] = novo[j], novo[i]
        elif tipo == "insertion":
            novo.insert(j, novo.pop(i))
        else:  # 2opt: inverte o trecho i..j
            novo[i:j + 1] = novo[i:j + 1][::-1]
        return novo

    # versoes publicas, para uso dos algoritmos
    def movimento_aleatorio(self, tipo=None):
        return self._movimento_aleatorio(tipo or self.vizinhanca)

    def aplicar(self, s, tipo, i, j):
        return self._aplicar(s, tipo, i, j)

    # ------------------------------------------------------------------
    # Avaliacao incremental
    #
    # Cache de uma solucao s: par (C, H), com k = 0..n
    #   C[k] = instante de conclusao apos atender os k primeiros ativos
    #   H[k] = atraso ponderado acumulado apos os k primeiros ativos
    # Se um movimento so altera as posicoes a partir de i (0-indexado), o
    # prefixo C[:i+1], H[:i+1] continua valido e so o sufixo e recalculado.
    # ------------------------------------------------------------------
    @staticmethod
    def posicao_inicial(tipo, i, j):
        """Primeira posicao alterada por um movimento (i, j)."""
        return min(i, j)

    def _estender(self, s, C, H, inicio):
        """Completa C e H (que ja trazem o prefixo 0..inicio) percorrendo s ate o fim."""
        t, p, d, w = self.t, self.p, self.d, self.w
        tempo, custo = C[inicio], H[inicio]
        atual = s[inicio - 1] if inicio > 0 else 0
        for pos in range(inicio, self.n):
            a = s[pos]
            tempo += t[atual][a] + p[a]
            atraso = tempo - d[a]
            if atraso > 0:
                custo += w[a] * atraso
            C.append(tempo)
            H.append(custo)
            atual = a

    def _cache_do_zero(self, s):
        C, H = [0], [0]
        self._estender(s, C, H, 0)
        return C, H

    def criar_cache(self, s):
        """Cache de s. Conta 1 avaliacao (equivale a h(s))."""
        self._contar()
        return self._cache_do_zero(s)

    @staticmethod
    def custo_do_cache(cache):
        return cache[1][-1]

    def atualizar_cache(self, s_novo, cache, inicio):
        """Cache de s_novo, reaproveitando o prefixo de cache ate `inicio`.
        Nao conta avaliacao. Use inicio = posicao_inicial(tipo, i, j)."""
        C, H = cache
        Cn, Hn = C[:inicio + 1], H[:inicio + 1]
        self._estender(s_novo, Cn, Hn, inicio)
        return Cn, Hn

    def h_swap(self, s, cache, i, j):
        """Custo do sucessor obtido trocando as posicoes i e j de s, sem criar a
        lista nova. Conta 1 avaliacao. `cache` e o cache de s."""
        if i > j:
            i, j = j, i
        self._contar()
        C, H = cache
        t, p, d, w = self.t, self.p, self.d, self.w
        tempo, custo = C[i], H[i]
        atual = s[i - 1] if i > 0 else 0
        for pos in range(i, self.n):
            if pos == i:
                a = s[j]
            elif pos == j:
                a = s[i]
            else:
                a = s[pos]
            tempo += t[atual][a] + p[a]
            atraso = tempo - d[a]
            if atraso > 0:
                custo += w[a] * atraso
            atual = a
        return custo

    def varredura_swap(self, s, cache):
        """Itera (i, j, custo) sobre toda a vizinhanca swap de s (conta 1 por sucessor)."""
        for i, j in self._movimentos("swap"):
            yield i, j, self.h_swap(s, cache, i, j)


# ----------------------------------------------------------------------
# Testes
# ----------------------------------------------------------------------
def _testar_instancia(caminho, nome):
    print(f"\n=== {nome} ({caminho}) ===")
    pb = ProblemaManutencao(caminho, seed=0)
    n = pb.n
    identidade = list(range(1, n + 1))
    print(f"n = {n}")
    print(f"h(identidade) = {pb.h(identidade)}")
    if nome == "pequena":
        assert pb._custo(identidade) == 4366, "h(identidade) deveria ser 4366"
        print("OK: h(identidade) = 4366")

    # contador: cada h conta 1
    pb.resetar_contador()
    for _ in range(10):
        pb.h(pb.solucao_aleatoria())
    assert pb.n_avaliacoes == 10, "contador incorreto"
    print("OK: contador conta 1 por chamada de h")

    # vizinhancas: validade, tamanho e ausencia de duplicatas
    for tipo in ProblemaManutencao.VIZINHANCAS:
        s = pb.solucao_aleatoria()
        if n <= 50:
            vizinhos = list(pb.sucessores(s, tipo))
            esperado = pb.tamanho_vizinhanca(tipo)
            assert len(vizinhos) == esperado, f"{tipo}: tamanho {len(vizinhos)} != {esperado}"
            assert len({tuple(v) for v in vizinhos}) == esperado, f"{tipo}: sucessores repetidos"
            assert all(v != s for v in vizinhos), f"{tipo}: sucessor igual a s"
            assert all(sorted(v) == identidade for v in vizinhos), f"{tipo}: permutacao invalida"
            print(f"OK: {tipo}: {esperado} sucessores distintos e validos")
        else:
            print(f"{tipo}: tamanho da vizinhanca = {pb.tamanho_vizinhanca(tipo)}")
        for _ in range(200):
            v = pb.sucessor_aleatorio(s, tipo)
            assert sorted(v) == identidade and v != s, f"{tipo}: sucessor aleatorio invalido"
    assert sorted(pb.mutacao(pb.solucao_aleatoria())) == identidade
    print("OK: sucessor_aleatorio e mutacao geram permutacoes validas")
    return pb


def _testar_incremental(pb):
    """A avaliacao incremental precisa concordar SEMPRE com a completa."""
    n = pb.n
    s = pb.solucao_aleatoria()
    cache = pb.criar_cache(s)
    assert pb.custo_do_cache(cache) == pb._custo(s), "cache inicial incorreto"

    # 1) h_swap contra avaliacao completa (bordas + sorteados)
    bordas = [(0, 1), (0, n - 1), (n - 2, n - 1), (0, n // 2), (n // 2, n - 1)]
    sorteados = [pb._movimento_aleatorio("swap") for _ in range(3000)]
    for i, j in bordas + sorteados:
        antes = pb.n_avaliacoes
        inc = pb.h_swap(s, cache, i, j)
        assert pb.n_avaliacoes == antes + 1, "h_swap deve contar 1 avaliacao"
        completo = pb._custo(pb._aplicar(s, "swap", i, j))
        assert inc == completo, f"swap ({i},{j}): incremental {inc} != completo {completo}"

    # 2) passeio aceitando todos os movimentos e atualizando o cache
    for _ in range(1000):
        i, j = pb._movimento_aleatorio("swap")
        novo = pb._aplicar(s, "swap", i, j)
        custo_vizinho = pb.h_swap(s, cache, i, j)
        antes = pb.n_avaliacoes
        cache = pb.atualizar_cache(novo, cache, pb.posicao_inicial("swap", i, j))
        assert pb.n_avaliacoes == antes, "atualizar_cache nao deve contar"
        s = novo
        assert cache == pb._cache_do_zero(s), "cache atualizado difere do cache do zero"
        assert pb.custo_do_cache(cache) == custo_vizinho

    # 2b) atualizar_cache tambem serve para insertion e 2opt
    for tipo in ("insertion", "2opt"):
        for _ in range(300):
            i, j = pb._movimento_aleatorio(tipo)
            novo = pb._aplicar(s, tipo, i, j)
            cache = pb.atualizar_cache(novo, cache, pb.posicao_inicial(tipo, i, j))
            s = novo
            assert cache == pb._cache_do_zero(s), f"{tipo}: cache atualizado incorreto"

    # 3) varredura completa da vizinhanca swap
    cont = 0
    for i, j, c in pb.varredura_swap(s, cache):
        assert c == pb._custo(pb._aplicar(s, "swap", i, j))
        cont += 1
    assert cont == pb.tamanho_vizinhanca("swap")
    print(f"OK: incremental == completo em {len(bordas) + len(sorteados)} swaps, "
          f"1000 passos com cache e varredura de {cont} sucessores")


def _medir_tempo(pb, repeticoes=20000):
    """Tempo por avaliacao de um sucessor swap: completa (aplicar + custo) vs incremental."""
    s = pb.solucao_aleatoria()
    cache = pb.criar_cache(s)
    movs = [pb._movimento_aleatorio("swap") for _ in range(repeticoes)]

    t0 = time.perf_counter()
    for i, j in movs:
        pb._custo(pb._aplicar(s, "swap", i, j))
    t_completo = (time.perf_counter() - t0) / repeticoes

    t0 = time.perf_counter()
    for i, j in movs:
        pb.h_swap(s, cache, i, j)
    t_incremental = (time.perf_counter() - t0) / repeticoes

    ganho = t_completo / t_incremental
    print(f"tempo por avaliacao: completa {t_completo * 1e6:.2f} us | "
          f"incremental {t_incremental * 1e6:.2f} us | ganho {ganho:.2f}x")
    return t_completo, t_incremental, ganho


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "."
    linhas = []
    for nome in ("pequena", "media", "grande"):
        caminho = os.path.join(pasta, f"manutencao_{nome}.txt")
        if not os.path.exists(caminho):
            continue
        pb = _testar_instancia(caminho, nome)
        _testar_incremental(pb)
        t_c, t_i, ganho = _medir_tempo(pb)
        linhas.append({
            "instancia": nome,
            "n": pb.n,
            "viz_swap": pb.tamanho_vizinhanca("swap"),
            "viz_insertion": pb.tamanho_vizinhanca("insertion"),
            "viz_2opt": pb.tamanho_vizinhanca("2opt"),
            "tempo_completa_us": round(t_c * 1e6, 3),
            "tempo_incremental_us": round(t_i * 1e6, 3),
            "ganho": round(ganho, 3),
        })
    if linhas:
        saida = os.path.join(pasta, "resultados_nucleo.csv")
        with open(saida, "w", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            escritor.writeheader()
            escritor.writerows(linhas)
        print(f"\nSalvo em {saida}")
    else:
        print(f"Nenhum manutencao_*.txt encontrado em '{pasta}'.")