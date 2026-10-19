# TP2 — Busca Local e Algoritmo Genético

Código do trabalho de Fundamentos de IA (UFMG) para sequenciamento de manutenção, com greedy descent, simulated annealing, algoritmo genético e avaliação incremental.

## Requisitos

Python 3.10+. `scipy` é necessário para os testes estatísticos do T2; `matplotlib`, para gerar gráficos do GA.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scipy matplotlib
```

## Como executar

Execute na raiz do repositório, onde estão as instâncias:

```bash
python problema.py .                # validações e medição incremental
python teste_incremental.py .       # testes da avaliação incremental e do greedy
python t2.py .                      # experimento completo do greedy
python t2.py . --rapido             # 3 execuções por configuração (teste reduzido)
python simulated_annealing.py .     # executa o SA
python algoritmo_genetico.py .      # executa o GA
python graficos_ga.py resultados_ga.csv --saida graficos_ga
```

O T2 usa 30 execuções nas instâncias pequena e média e 10 na grande, com orçamentos de 50.000, 150.000 e 300.000 avaliações. `--rapido` reduz as repetições, não o orçamento.

## Saídas

- `problema.py`: `resultados_nucleo.csv`.
- `t2.py`: `resultados_t2.csv`, `resultados_t2_execucoes.csv` e, com SciPy, `resultados_t2_testes.csv`.
- `algoritmo_genetico.py`: `resultados_ga.csv` e `resumo_ga.csv`.
- `graficos_ga.py`: gráficos e séries agregadas na pasta indicada.
- O SA imprime resumos no terminal; a versão atual não salva CSVs nem gráficos.

Os scripts podem sobrescrever arquivos de resultados existentes. Faça cópias antes de executar novamente. O teste incremental é executado diretamente por Python e não requer `pytest`.
