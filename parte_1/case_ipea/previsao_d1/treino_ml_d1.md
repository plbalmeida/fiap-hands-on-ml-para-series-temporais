# Treino do modelo de ML com previsão D+1

Agora um modelo de ML de regressão supervisionado será implementado para previsão do preço de pretróleo em D+1, seguindo boas práticas para o treino de modelo de ML para série temporal:

- **Divisão de Dados**: Dividir os dados em conjuntos de treino e teste de forma sequencial para preservar a ordem temporal.

- **Normalização**: Escalar os dados para melhorar a performance do modelo.

- **Validação Cruzada**: Utilizar técnicas como validação cruzada em blocos para avaliar a performance do modelo em diferentes períodos de tempo.

## Divisão dos dados

A ordem dos dados importa quando se trata de série temporal, por isso não se deve usar a função `train_test_split` do sci-kit learning para dividir os dados, pois ela faz isso randomicamente.

A seguir, a divisão do conjunto de dados original é feita obedecendo a ordem dos dados, sendo 90% para o conjunto de dados de treino e 10% para o conjunto de dados de teste.

```python
# divisão manual dos dados de treino e teste
train_size = int(len(eia366) * 0.9)
train, test = eia366.iloc[:train_size], eia366.iloc[train_size:]
```

Na sequência é implementada a classe `FeatureEngineer` para que a engenharia de features seja feita com a classe `Pipeline` do sci-kit learn, ainda no `Pipeline` os dados são normalizados com a classe `StandardScaler`, e é usada classe `GradientBoostingRegressor` como modelos para treino e teste.

Para validação cruzada no treino do modelo, será utilizada a classe `TimeSeriesSplit` que lida com dados de série temporal para essa tarefa.

<div align="center">
  <figure>
    <img src="ts_split.png" alt="Divisão dos Dados Temporais">
    <figcaption>
      Fonte: <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py">Scikit-learn - Cross-validation indices</a>
    </figcaption>
  </figure>
</div>

E é configurado do `HalvingGridSearchCV` para busca da melhor combinação de hiperparâmetros. O `HalvingGridSearchCV` pode ser uma opção melhor que o clássico `GridSearchCV` quando é necessário otimizar hiperparâmetros de forma eficiente e escalável, especialmente em cenários com muitos hiperparâmetros e grandes volumes de dados, onde a busca exaustiva do `GridSearchCV` seria impraticável. Documentação do `HalvingGridSearchCV`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html)

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# classe customizada para engenharia de features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target, lags, window_size):
        self.target = target
        self.lags = lags
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in range(1, self.lags + 1):
            X[f"lag_{lag}"] = X[self.target].shift(lag)
        X[f"rolling_mean_{self.window_size}"] = X[self.target].shift(1).rolling(window=self.window_size).mean()
        X["diff"] = X[self.target].shift(1).diff()
        X["month"] = X.index.month
        X["day_of_week"] = X.index.dayofweek
        X[f"rolling_std_{self.window_size}"] = X[self.target].shift(1).rolling(window=self.window_size).std()
        X["day"] = X.index.day
        X["quarter"] = X.index.quarter
        X["year"] = X.index.year
        X = X.drop(columns=["value_usd"])
        X.fillna(0, inplace=True)
        return X

# pipeline de steps
pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer(target="value_usd", lags=7, window_size=7)),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor())
])

# espaço amostral de hiperparâmetros
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1]
}

# TimeSeriesSplit para validação cruzada
tscv = TimeSeriesSplit(n_splits=3)

# HalvingGridSearchCV para busca de melhor combinação de hiperparâmetros
search = HalvingGridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    factor=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1
)

X = train.copy()  # inclui a coluna 'value_usd' no X para uso no pipeline de feature engineering
y = train["value_usd"]

# fit do modelo
search.fit(X, y)

# melhores hiperparâmetros e score
print("Best parameters found: ", search.best_params_)
print("Best score: ", search.best_score_)
```