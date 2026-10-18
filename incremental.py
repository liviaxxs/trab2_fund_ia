import itertools


def h_inc(pb, s, cache, tipo, i, j):
    """Custo do sucessor (tipo, i, j) de s, sem criar a lista inteira.
    Conta 1 avaliação. `cache` é o cache de s (par C, H)."""
    lo, hi = (i, j) if i < j else (j, i)

    # trecho lo..hi como ficaria depois do movimento
    if tipo == "swap":
        seg = s[lo:hi + 1]
        seg[0], seg[-1] = seg[-1], seg[0]
    elif tipo == "2opt":
        seg = s[lo:hi + 1][::-1]
    else:  # insertion: tira s[i] e insere na posição j
        if i < j:
            seg = s[i + 1:j + 1] + [s[i]]
        else:
            seg = [s[i]] + s[j:i]

    pb._contar()
    C, H = cache
    t, p, d, w = pb.t, pb.p, pb.d, pb.w
    tempo, custo = C[lo], H[lo]                 # prefixo antes de lo não muda
    atual = s[lo - 1] if lo > 0 else 0
    for a in itertools.chain(seg, s[hi + 1:]):  # trecho alterado + resto igual
        tempo += t[atual][a] + p[a]
        atraso = tempo - d[a]
        if atraso > 0:
            custo += w[a] * atraso
        atual = a
    return custo