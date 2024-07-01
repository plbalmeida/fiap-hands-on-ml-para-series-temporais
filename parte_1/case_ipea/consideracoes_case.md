# Considerações sobre o case do IPEA

Algumas técnicas podem ser necessárias para melhorar o código apresentado nesse case, como aumentar o espaço amostral de hiperparâmetros, considerar incluir uma etapa de seleção de features (em caso de conjunto de dados hiperdimensional), etc.

A seguir essas sugestões serão discutidas.

1) Aumento do espaço amostral de hiperparâmetros

Para explorar mais combinações de hiperparâmetros e possivelmente encontrar um modelo com melhor desempenho, podemos ampliar o espaço amostral de hiperparâmetros. Por exemplo:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_depth": [3, 5, 7],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}
```

2) Método de seleção de features

Em casos de conjuntos de dados com muitas features (hiperdimensionais), a seleção de features pode melhorar o desempenho e reduzir o overfitting. Podemos usar `SelectKBest` ou `RFE` como parte do pipeline. Exemplo:

```python
from sklearn.feature_selection import SelectKBest, f_regression

pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer(target="value_usd", lags=7, window_size=7)),
    ("feature_selection", SelectKBest(score_func=f_regression, k=10)),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor())
])
```

3) Testar outros modelos

É interessante avaliar a performance de outros modelos para a previsão do fenômeno de interesse, inclusive modelos clássicos como ARIMA ([https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)), por exemplo, e utilizar o mesmo como *base line* quando não se tem um ponto de refência com relação a performance de previsão.

4) Incluir mais variáveis

Pode ser útil incluir mais variáveis a partir da variável dependente original, ou incluir variáveis exógenas afim de melhorar a performance preditiva dos modelos, a decisão de incluir mais variáveis depende do domínio do Analista sobre o fenâmeno que está sendo modelado.
