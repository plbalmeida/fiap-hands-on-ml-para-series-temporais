## Avaliação de Performance de ML para previsão D+15

### Código para Avaliação

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error, 
    r2_score
)

# extraindo as features e o target
X_test = test.copy()
y_test = pd.concat([test["value_usd"].shift(-i) for i in range(1, 16)], axis=1).dropna()
y_test.columns = [f"value_usd_t{i}" for i in range(1, 16)]
X_test = X_test.iloc[:len(y_test)]  # alinhando X_test e y_test

# previsões no conjunto de teste
y_pred = search.predict(X_test)

# avaliação da performance
mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
mape = mean_absolute_percentage_error(y_test, y_pred, multioutput="raw_values")
r2 = r2_score(y_test, y_pred, multioutput="raw_values")

print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)
print("R² Score:", r2)

Mean Absolute Error: [2.32440746 2.75457208 3.10289434 3.4246487  3.69572298 3.9503334
 4.19873355 4.38715484 4.58190056 4.84711349 5.03588346 5.2700822
 5.54430392 5.76858427 5.96091443]
Mean Absolute Percentage Error: [0.03573954 0.04229321 0.04819819 0.05373746 0.058147   0.0628686
 0.06801432 0.07184996 0.07550796 0.07995469 0.08391442 0.0880025
 0.09232921 0.09597728 0.09959782]
R² Score: [0.97690324 0.96965349 0.96157372 0.95358242 0.94678207 0.94017205
 0.93490549 0.92973337 0.92328849 0.91573162 0.91084769 0.90419681
 0.89474895 0.88513475 0.87760761]
```

Agora cada output é uma lista com a performance de cada modelo para cada *D+n*, é poss´´ivel avaliar performance em um data frame.

```python
pd.DataFrame({
    "Previsão": [f"D+{i}" for i in range(1, 16)],
    "Mean Absolute Error": mae,
    "Mean Absolute Percentage Error": mape,
    "R² Score": r2
})
```

É posssível perceber que para quase todas as métricas, os modelos pioram a performance ao avançar em *D+n*.

A seguir, o código permite visualizar as previsões contra o conjunto de dados de teste para algumas datas.

```python
import matplotlib.pyplot as plt

dates = y_test.index

for i in range(0, len(dates), 50):
    plt.figure(figsize=(10, 5))
    plt.title("Série Temporal - EIA366_PBRENT366")
    plt.plot(range(1, 16), y_test.iloc[i, :], label=f"Conjunto de teste")
    plt.plot(range(1, 16), y_pred[i, :], label=f"Previsões D+15, data inicial: {dates[i].strftime('%Y-%m-%d')}", linestyle="--")
    plt.xlabel("Dia")
    plt.ylabel("US$")
    plt.legend()
    plt.xticks(range(1, 16))
    plt.show()
```

