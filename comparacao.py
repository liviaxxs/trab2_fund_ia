import random
import matplotlib.pyplot as plt

# ==========================================
# 1. ESTRUTURA DE DADOS E FUNÇÃO DE AVALIAÇÃO
# ==========================================

class Instancia:
    def __init__(self, n, p, w, d):
        self.n = n  # número de trabalhos/ativos
        self.p = p  # tempos de processamento
        self.w = w  # pesos de atraso
        self.d = d  # prazos de entrega (due dates)

def gerar_instancia(n, seed=42):
    random.seed(seed)
    p = [random.randint(1, 20) for _ in range(n)]
    w = [random.randint(1, 10) for _ in range(n)]
    # Due dates proporcionais à carga de trabalho
    sum_p = sum(p)
    d = [random.randint(int(sum_p * 0.1), int(sum_p * 0.6)) for _ in range(n)]
    return Instancia(n, p, w, d)

def avaliar(pi, instancia):
    """
    Calcula o Atraso Ponderado Total h(pi) para uma permutação pi.
    pi é uma lista de índices (0 a n-1).
    """
    tempo_atual = custo_total = 0
    for tarefa in pi:
        tempo_atual += instancia.p[tarefa]
        atraso = max(0, tempo_atual - instancia.d[tarefa])
        custo_total += instancia.w[tarefa] * atraso
    return custo_total


# ==========================================
# 2. OPERADORES DE VIZINHANÇA
# ==========================================

def get_neighbors_swap(pi):
    vizinhos = []
    n = len(pi)
    for i in range(n):
        for j in range(i + 1, n):
            viz = pi.copy()
            viz[i], viz[j] = viz[j], viz[i]
            vizinhos.append(viz)
    return vizinhos

def get_neighbors_insertion(pi):
    vizinhos = []
    n = len(pi)
    for i in range(n):
        element = pi[i]
        rest = pi[:i] + pi[i+1:]
        for j in range(len(rest) + 1):
            viz = rest[:j] + [element] + rest[j:]
            if viz != pi:
                vizinhos.append(viz)
    return vizinhos

def get_neighbors_2opt(pi):
    vizinhos = []
    n = len(pi)
    for i in range(n):
        for j in range(i + 2, n + 1):
            viz = pi.copy()
            viz[i:j] = reversed(viz[i:j])
            if viz != pi:
                vizinhos.append(viz)
    return vizinhos


# ==========================================
# 3. ALGORITMOS DE OTIMIZAÇÃO
# ==========================================

def greedy_random_restart(instancia, max_evals=10000):
    """Greedy Descent com Random Restart"""
    melhor_solucao_global = None
    melhor_custo_global = float('inf')
    
    avaliacoes_feitas = 0
    historico_convergencia = []
    
    while avaliacoes_feitas < max_evals:
        # Restart aleatório
        pi_atual = list(range(instancia.n))
        random.shuffle(pi_atual)
        custo_atual = avaliar(pi_atual, instancia)
        avaliacoes_feitas += 1
        
        if custo_atual < melhor_custo_global:
            melhor_custo_global = custo_atual
            melhor_solucao_global = pi_atual.copy()
            
        historico_convergencia.append((avaliacoes_feitas, melhor_custo_global))
        
        melhoria = True
        while melhoria and avaliacoes_feitas < max_evals:
            melhoria = False
            # Gerar vizinhança (utilizando insertion por padrão por ser eficiente)
            vizinhos = get_neighbors_insertion(pi_atual)
            
            melhor_vizinho = None
            melhor_custo_vizinho = custo_atual
            
            for viz in vizinhos:
                if avaliacoes_feitas >= max_evals:
                    break
                custo_viz = avaliar(viz, instancia)
                avaliacoes_feitas += 1
                
                if custo_viz < melhor_custo_vizinho:
                    melhor_custo_vizinho = custo_viz
                    melhor_vizinho = viz
            
            if melhor_vizinho is not None and melhor_custo_vizinho < custo_atual:
                pi_atual = melhor_vizinho
                custo_atual = melhor_custo_vizinho
                melhoria = True
                
                if custo_atual < melhor_custo_global:
                    melhor_custo_global = custo_atual
                    melhor_solucao_global = pi_atual.copy()
            
            historico_convergencia.append((avaliacoes_feitas, melhor_custo_global))
            
    return melhor_custo_global, historico_convergencia


def simulated_annealing(instancia, max_evals=10000, T0=100.0, Tf=0.01, alpha=0.95):
    """Simulated Annealing"""
    pi_atual = list(range(instancia.n))
    random.shuffle(pi_atual)
    custo_atual = avaliar(pi_atual, instancia)
    avaliacoes_feitas = 1
    
    melhor_solucao = pi_atual.copy()
    melhor_custo = custo_atual
    
    historico_convergencia = [(avaliacoes_feitas, melhor_custo)]
    
    T = T0
    while avaliacoes_feitas < max_evals and T > Tf:
        # Gerar um vizinho aleatório (ex: swap ou insertion)
        n = instancia.n
        i, j = random.sample(range(n), 2)
        pi_viz = pi_atual.copy()
        pi_viz[i], pi_viz[j] = pi_viz[j], pi_viz[i]
        
        custo_viz = avaliar(pi_viz, instancia)
        avaliacoes_feitas += 1
        
        delta = custo_viz - custo_atual
        if delta < 0 or random.random() < __import__('math').exp(-delta / T):
            pi_atual = pi_viz
            custo_atual = custo_viz
            
            if custo_atual < melhor_custo:
                melhor_custo = custo_atual
                melhor_solucao = pi_atual.copy()
                
        historico_convergencia.append((avaliacoes_feitas, melhor_custo))
        T *= alpha
        
    # Se sobrar orçamento, preenche com o melhor atual
    while avaliacoes_feitas < max_evals:
        avaliacoes_feitas += 1
        historico_convergencia.append((avaliacoes_feitas, melhor_custo))
        
    return melhor_custo, historico_convergencia


def genetic_algorithm(instancia, max_evals=10000, pop_size=50, taxa_mutacao=0.2):
    """Algoritmo Genético com Order Crossover (OX)"""
    def crossover(pai1, pai2):
        n = len(pai1)
        a, b = sorted(random.sample(range(n), 2))
        filho = [None] * n
        filho[a:b+1] = pai1[a:b+1]
        
        ptr = (b + 1) % n
        for gene in pai2[b+1:] + pai2[:b+1]:
            if gene not in filho:
                filho[ptr] = gene
                ptr = (ptr + 1) % n
        return filho

    def mutacao(pi):
        if random.random() < taxa_mutacao:
            i, j = random.sample(range(len(pi)), 2)
            pi[i], pi[j] = pi[j], pi[i]
        return pi

    # População inicial
    population = [list(range(instancia.n)) for _ in range(pop_size)]
    for ind in population:
        random.shuffle(ind)
        
    fitness = [avaliar(ind, instancia) for ind in population]
    avaliacoes_feitas = pop_size
    
    melhor_custo_global = min(fitness)
    historico_convergencia = [(avaliacoes_feitas, melhor_custo_global)]
    
    while avaliacoes_feitas < max_evals:
        nova_pop = []
        # Elitismo: mantém o melhor
        best_idx = fitness.index(min(fitness))
        nova_pop.append(population[best_idx].copy())
        
        while len(nova_pop) < pop_size and avaliacoes_feitas < max_evals:
            # Seleção por torneio binário
            p1 = min(random.sample(range(pop_size), 2), key=lambda x: fitness[x])
            p2 = min(random.sample(range(pop_size), 2), key=lambda x: fitness[x])
            
            filho = crossover(population[p1], population[p2])
            filho = mutacao(filho)
            
            custo_filho = avaliar(filho, instancia)
            avaliacoes_feitas += 1
            
            nova_pop.append(filho)
            if custo_filho < melhor_custo_global:
                melhor_custo_global = custo_filho
                
            historico_convergencia.append((avaliacoes_feitas, melhor_custo_global))
            
        population = nova_pop
        fitness = [avaliar(ind, instancia) for ind in population]
        avaliacoes_feitas += len(population)

    return melhor_custo_global, historico_convergencia


# ==========================================
# 4. EXPERIMENTOS E GERAÇÃO DOS GRÁFICOS
# ==========================================

def run_experiments_and_plot():
    n_jobs = 50
    max_evals = 10000
    num_runs = 15
    
    print(f"Gerando instância com n = {n_jobs} trabalhos...")
    instancia = gerar_instancia(n_jobs, seed=42)
    
    algoritmos = {
        'Greedy Restart': greedy_random_restart,
        'Simulated Annealing': simulated_annealing,
        'Algoritmo Genético': genetic_algorithm
    }
    
    resultados_finais = {alg: [] for alg in algoritmos}
    historicos_completos = {alg: [] for alg in algoritmos}
    
    print(f"Executando {num_runs} repetições para cada algoritmo (Orçamento: {max_evals} avaliações)...")
    for run in range(num_runs):
        for nome, func in algoritmos.items():
            custo, hist = func(instancia, max_evals=max_evals)
            if run == 0:
                historicos_completos[nome] = hist
            resultados_finais[nome].append(custo)
        print(f"  Corrida {run+1}/{num_runs} concluída.")
        
    # Plotando os Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Curva de Convergência Média (usando a primeira corrida como representativa do perfil)
    for nome, hist in historicos_completos.items():
        x = [h[0] for h in hist]
        y = [h[1] for h in hist]
        ax1.plot(x, y, label=nome, linewidth=2)
        
    ax1.set_title(r'Curva de Convergência ao Longo das Avaliações', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Número de Avaliações da Função Objetivo', fontsize=10)
    ax1.set_ylabel(r'Custo Médio ($h(\pi)$)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # 2. Boxplot Comparativo
    labels = list(algoritmos.keys())
    data_to_plot = [resultados_finais[alg] for alg in labels]
    
    bp = ax2.boxplot(data_to_plot, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    
    cores = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], cores):
        patch.set_facecolor(color)
        
    # Configuração correta de ticks para compatibilidade com qualquer versão do Matplotlib
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontsize=10)
    
    ax2.set_title(r'Distribuição dos Custos Finais (15 Execuções)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Algoritmos', fontsize=10)
    ax2.set_ylabel(r'Atraso Ponderado Total ($h(\pi)$)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('graficos_comparativos.png', dpi=300)
    print("\Gráficos salvos com sucesso no arquivo 'graficos_comparativos.png'!")
    plt.show()

if __name__ == '__main__':
    run_experiments_and_plot()