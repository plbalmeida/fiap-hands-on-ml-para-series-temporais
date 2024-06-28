## Avaliação de Performance de ML para previsão D+1

A avaliação da performance de modelos de ML para séries temporais envolve o uso de métricas específicas para medir a precisão das previsões. Algumas das métricas comuns incluem:

- **Erro Quadrático Médio (MSE)**: Média dos quadrados das diferenças entre os valores previstos e os valores reais.

- **Erro Absoluto Médio (MAE)**: Média das diferenças absolutas entre os valores previstos e os valores reais.

- **Erro Absoluto Médio Percentual (MAPE)**: Média percentual das diferenças absolutas entre os valores previstos e os valores reais.

- **Coeficiente de Determinação (R²)**: Medida da proporção da variância nos dados que é explicada pelo modelo.

### Código para Avaliação

```python
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error, 
    r2_score
)

# extraindo as features e o target
X_test = test.copy()
y_test = test["value_usd"]

# previsões no conjunto de teste
y_pred = search.predict(X_test)

# avaliação da performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)
print("R² Score:", r2)
```

E é possível checar a visualmente as previsões e o conjunto de ddos de teste com o `matplotlib`.

```python
# plot das previsões vs valores reais
plt.figure(figsize=(10, 5))
plt.title("Série Temporal - EIA366_PBRENT366")
plt.plot(y_test.index, y_test, label="Conjunto de teste")
plt.plot(y_test.index, y_pred, label="Previsões D+1", linestyle='--')
plt.xlabel("Data")
plt.ylabel("US$")
plt.legend()
plt.show()
```