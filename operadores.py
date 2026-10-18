def swap(s, i, j):
    """Swap clássico, recebe o array original e as posições a serem trocadas i e j 
        retorna o array t com as posições modificadas"""
    
    t = s.copy()
    t[i] = s[j]
    t[j] = s[i]

    return t

def insertion(s, i, j):
    """insertion clássico, recebe o array original e onde o valor da posição i para ser inserido em j"""
    t = s.copy()
    novo = t.pop(i)
    t.insert(j, novo)
    return t

def two_opt(s, i, j):
    """inverte a posição das casas de i até j"""
    t = s.copy()
    while i < j: 
        t[i] = s[j]
        t[j] = s[i]
        i += 1
        j -= 1

    return t

            
            

