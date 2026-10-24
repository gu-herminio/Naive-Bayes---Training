from sklearn.naive_bayes import BernoulliNB

# ----------------------------------
# 1. Nossa base de exemplos
# ----------------------------------

# Cada dia tem duas características:
# X = [está ensolarado?, está quente?]
#
# 0 = Não
# 1 = Sim

X = [
    [1, 1],  # ensolarado e quente
    [1, 1],
    [1, 0],  # ensolarado, mas não quente
    [0, 1],  # não ensolarado, mas quente
    [0, 0],  # não ensolarado e não quente
    [0, 0],
    [1, 1],
    [0, 1],
]

# y é o resultado real de cada dia:
# 1 = Praia
# 0 = Não praia

y = [
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    0,
]

# ----------------------------------
# 2. Criar e treinar o modelo
# ----------------------------------

modelo = BernoulliNB()

# O modelo aprende com os exemplos de X e suas respostas em y
modelo.fit(X, y)

# ----------------------------------
# 3. Fazer uma previsão para um dia novo
# ----------------------------------

# Novo dia: ensolarado e quente
dia_novo = [[1, 0]]

previsao = modelo.predict(dia_novo)[0]
probabilidades = modelo.predict_proba(dia_novo)[0]

# Traduzir o número previsto para uma palavra
if previsao == 1:
    resultado = "Praia"
else:
    resultado = "Não praia"

print("Previsão:", resultado)

print(f"Probabilidade estimada de Não praia: {probabilidades[0]:.1%}")
print(f"Probabilidade estimada de Praia:     {probabilidades[1]:.1%}")
