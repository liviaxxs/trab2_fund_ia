
import pandas as pd
from scipy.stats import mannwhitneyu

dados = pd.read_csv("resultados_ga.csv")

dados = dados.sort_values("avaliacoes")

finais = (
    dados.groupby(
        ["instancia", "experimento", "rodada"],
        as_index=False
    )
    .tail(1)
)

print("Quantidade de execuções identificadas:")
print(
    finais.groupby(["instancia", "experimento"])
    .size()
)

print("\nTestes de Mann–Whitney (bilaterais):")

for instancia in ["pequena", "media", "grande"]:
    parte = finais[finais["instancia"] == instancia]

    ox = parte.loc[
        parte["experimento"] == "OX", "melhor_h"
    ].dropna()

    sem_cruzamento = parte.loc[
        parte["experimento"] == "sem_cruzamento", "melhor_h"
    ].dropna()

    if len(ox) < 2 or len(sem_cruzamento) < 2:
        print(f"\n{instancia}: dados insuficientes")
        continue

    resultado = mannwhitneyu(
        ox,
        sem_cruzamento,
        alternative="two-sided"
    )

    print(f"\nInstância: {instancia}")
    print(f"Execuções OX: {len(ox)}")
    print(f"Execuções sem cruzamento: {len(sem_cruzamento)}")
    print(f"Mediana OX: {ox.median():.2f}")
    print(f"Mediana sem cruzamento: {sem_cruzamento.median():.2f}")
    print(f"U = {resultado.statistic}")
    print(f"p-valor = {resultado.pvalue:.8g}")
