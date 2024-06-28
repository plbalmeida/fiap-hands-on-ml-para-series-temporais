## Avaliação de Performance de ML para Série Temporal

A avaliação da performance de modelos de ML para séries temporais envolve o uso de métricas específicas para medir a precisão das previsões. Algumas das métricas comuns incluem:

- **Erro Absoluto Médio (MAE)**: Média das diferenças absolutas entre os valores previstos e os valores reais.
- **Erro Quadrático Médio (MSE)**: Média dos quadrados das diferenças entre os valores previstos e os valores reais.
- **Raiz do Erro Quadrático Médio (RMSE)**: Raiz quadrada do MSE, útil para interpretar os erros na mesma escala dos dados.
- **Coeficiente de Determinação (R²)**: Medida da proporção da variância nos dados que é explicada pelo modelo.

### Exemplo de Código para Avaliação

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Supomos que y_test e y_pred sejam os valores reais e previstos, respectivamente
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
```