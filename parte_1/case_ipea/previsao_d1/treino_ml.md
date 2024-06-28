# Previsão de Série Temporal com ML

Práticas recomendadas para o treino de modelo de ML para série temporal:

- **Divisão de Dados**: Dividir os dados em conjuntos de treino e teste de forma sequencial para preservar a ordem temporal.
- **Normalização**: Escalar os dados para melhorar a performance do modelo.
- **Validação Cruzada**: Utilizar técnicas como validação cruzada em blocos para avaliar a performance do modelo em diferentes períodos de tempo.
- **Seleção de Modelos**: Experimentar com diferentes modelos de ML, como Regressão Linear, Árvores de Decisão, Random Forest e Redes Neurais.

## Divisão dos dados

A ordem dos dados importa quando se trata de série temporal, por isso não se deve usar a função `train_test_split` do sci-kit learning para dividir os dados, pois ela faz isso randomicamente.

A seguir, a divisão do conjunto de dados original é feita obedecendo a ordem dos dados, sendo 80% para o conjunto de dados de treino e 20% para o conjunto de dados de teste.

```python
# divisão manual dos dados de treino e teste
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
```

Na sequência os dados são separados entra o conjuntode dados de variáveis indenpendetes e dependente, de treino e teste respectivamente.

```python
# definindo as variáveis independentes e dependente
X_train, y_train = train.drop(columns=["value_usd"]), train["value_usd"]
X_test, y_test = test.drop(columns=["value_usd"]), test["value_usd"]
```

