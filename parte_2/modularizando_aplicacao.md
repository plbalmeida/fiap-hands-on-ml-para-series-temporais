# Modularizando a Aplicação

Modularizar um código traz inúmeras vantagens, incluindo a melhoria da organização e manutenção do código. Ao dividir o código em módulos menores e mais gerenciáveis, cada um com uma responsabilidade específica, facilita-se a leitura e a compreensão do código por diferentes desenvolvedores. Além disso, a modularização promove a reutilização de código, permitindo que funções ou classes desenvolvidas para um módulo possam ser facilmente utilizadas em outros projetos ou partes do mesmo projeto, economizando tempo e esforço.

A testabilidade também é aprimorada, pois é mais simples escrever e executar testes unitários para pequenos módulos independentes do que para um grande bloco de código monolítico. Por fim, a modularização facilita a identificação e correção de bugs, contribuindo para um desenvolvimento mais eficiente e uma melhor qualidade de software.

Módulo 

A classe `FeatureEngineer` será modularizada, assim como será criada uma função para treino do modelo, e uma função para preparar o conjunto de dados de target.

Primeiro é criado o diretório `src/`, depois o `__init__.py` para modularizar os scripts do diretório, e o `feature_engineer.py` para colocar a classe de features engineer:

```bash
mkdir src
echo src/__init__.py
echo src/feature_engineer.py
```

> O arquivo `__init__.py` é utilizado para indicar que o diretório onde ele está presente deve ser tratado como um pacote do python. Em outras palavras, ele permite que os módulos dentro desse diretório sejam importados como parte de um pacote.

```python
# src/feature_engineer.py

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer para séries temporais.

    Esta classe cria várias características baseadas em uma série temporal,
    como lags, médias móveis, diferenças etc.

    Args:
        target (str): O nome da coluna alvo na série temporal.
        lags (int): O número de defasagens a serem criadas.
        window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.

    Attributes:
        target (str): O nome da coluna alvo na série temporal.
        lags (int): O número de defasagens a serem criadas.
        window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.
    """  # noqa

    def __init__(self, target, lags, window_size):
        """
        Inicializa o FeatureEngineer com os parâmetros fornecidos.

        Args:
            target (str): O nome da coluna alvo na série temporal.
            lags (int): O número de defasagens a serem criadas.
            window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.
        """  # noqa
        self.target = target
        self.lags = lags
        self.window_size = window_size

    def fit(self, X, y=None):
        """
        Método de ajuste necessário para conformidade com o scikit-learn,
        não realiza nenhuma operação.

        Args:
            X (pd.DataFrame): O dataframe de entrada.
            y (pd.Series, opcional): A série alvo (não utilizada).

        Returns:
            self: Retorna a instância do próprio objeto.
        """
        return self

    def transform(self, X):
        """
        Transforma a série temporal adicionando características engenheiradas.

        Args:
            X (pd.DataFrame): O dataframe de entrada contendo a série temporal.

        Returns:
            pd.DataFrame: Um novo dataframe com as características adicionadas.
        """
        X = X.copy()

        for lag in range(0, self.lags):
            X[f"lag_{lag+1}"] = X[self.target].shift(lag)

        for window in self.window_size:
            X[f"rolling_mean_{window}"] = X[self.target].rolling(window=window).mean()  # noqa
            X[f"rolling_std_{window}"] = X[self.target].rolling(window=window).std()  # noqa
            X[f"ewm_mean_{window}"] = X[self.target].ewm(span=window).mean()
            X[f"ewm_std_{window}"] = X[self.target].ewm(span=window).std()

        X["diff"] = X[self.target].diff()
        X["year"] = X.index.year
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["day"] = X.index.day
        X["day_of_week"] = X.index.dayofweek
        X = X.drop(columns=[self.target])
        X.fillna(0, inplace=True)
        return X

```

> Repare que a classe e métodos possuem docstring do tipo Google style: [https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html)

Agora vamos modularizar a função do modelo de ML:

```bash
echo > src/model_train.py
```

```python
# src/model_train.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.feature_engineer import FeatureEngineer


def create_pipeline_and_search(target, lags, window_size):
    """
    Cria uma pipeline e configura uma busca de hiperparâmetros usando HalvingGridSearchCV.

    Args:
        target (str): Nome da coluna alvo.
        lags (int): Número de lags a serem criados.
        window_size (int): Janela para criar features baseadas em janelas.

    Returns:
        HalvingGridSearchCV: Objeto configurado para busca de hiperparâmetros.
    """  # noqa 401
    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer(target, lags, window_size)),  # noqa 401
        ("scaler", StandardScaler()),
        ("model", RegressorChain(base_estimator=GradientBoostingRegressor(random_state=123), random_state=123))  # noqa
    ])

    param_grid = {
        "model__base_estimator__n_estimators": [100, 200, 300],
        "model__base_estimator__learning_rate": [0.01, 0.05, 0.1],
        "model__base_estimator__max_depth": [3, 5, 8],
        "model__base_estimator__min_samples_split": [2, 5, 10],
        "model__base_estimator__min_samples_leaf": [1, 2, 4]
    }

    tscv = TimeSeriesSplit(n_splits=6)

    search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        factor=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=132
    )

    return search

```

E a função para transformar o conjunto de dados de target:

```bash
echo > src/utils.py
```

```python
# src/utils.py

import pandas as pd


def target_transform(train, target, horizon):
    """
    Transforma a coluna alvo para prever múltiplos passos à frente.

    Concatena colunas deslocadas da coluna alvo para criar um DataFrame que
    contém os valores da coluna alvo para múltiplos passos à frente, até o horizonte
    especificado.

    Args:
        train (pd.DataFrame): O DataFrame de treino contendo a(s) coluna(s) de dados.
        target (str): O nome da coluna alvo que se deseja transformar.
        horizon (int): O número de passos à frente que se deseja prever.

    Returns:
        pd.DataFrame: Um DataFrame contendo as colunas da coluna alvo deslocadas
        para prever múltiplos passos à frente. As colunas são nomeadas no formato 
        `target_t{i+1}`, onde `i` é o número do passo.

    Example:
        >>> train = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> target_transform(train, 'target', 3)
           target_t1  target_t2  target_t3
        0        2.0        3.0        4.0
        1        3.0        4.0        5.0
        2        4.0        5.0        6.0
        3        5.0        6.0        7.0
        4        6.0        7.0        8.0
        5        7.0        8.0        9.0
        6        8.0        9.0       10.0
    """  # noqa 401
    y = pd.concat([train[target].shift(-i) for i in range(0, horizon)], axis=1).dropna()  # noqa 401
    y.columns = [f"{target}_t{i+1}" for i in range(0, horizon)]
    return y

```