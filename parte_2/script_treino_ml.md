# Script para Treino e Previsões

Agora criamos o script para execução do treino de ML e previsões:

```bash
mkdir model_training
echo > model_training/main.py
mkdir streamlit  # path onde será salvo os outputs que serão consumidos pelo Streamlit 
```

Script `main.py`:

```python
# model_training/main.py

import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import ipeadatapy as ip  # noqa
import pandas as pd  # noqa
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
from pandas.tseries.offsets import BDay  # noqa
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error  # noqa
from model_train import create_pipeline_and_search  # noqa
from src.utils import target_transform  # noqa


def main():

    print("\n--------- Variáveis ---------\n")

    horizon = 15  # horizonte de previsão
    target = "value_usd"  # variável target
    lags = 15  # quantidade de lags
    window_size = [3, 7, 15]  # janela para média e desvio padrão móvel
    years = 5  # tamanho do conjunto de dados em anos

    print(f"Variável target: {target}")
    print(f"Horizonte de previsão: {horizon}")  # noqa
    print(f"Lags: {lags}")
    print(f"Janela p/ média e desvio padrão móvel: {window_size}")
    print(f"Tamanho do conjunto de dados em anos: {years}")

    # preço do petróleo bruto
    cod = "EIA366_PBRENT366"
    eia366 = ip.timeseries(cod)

    # tamanho do conjunto de dados
    years_ago = pd.Timestamp.today() - pd.DateOffset(years=years)
    eia366 = eia366.loc[years_ago:]
    eia366 = eia366[["VALUE (US$)"]]
    eia366.rename(columns={"VALUE (US$)": "value_usd"}, inplace=True)
    eia366.index.name = "date"
    eia366 = eia366.dropna()

    # divisão manual dos dados de treino e teste
    train_size = int(len(eia366) * 0.90)
    train, test = eia366.iloc[:train_size], eia366.iloc[train_size:]

    # inclui a coluna target para feature engineering
    X_train = train.copy()

    # conjunto de target dado o horizonte de previsão
    y_train = target_transform(
        train,
        target,
        horizon
    )

    # alinhando X e y
    X_train = X_train.iloc[:len(y_train)]

    print("\n--------- Treino do modelo ---------\n")

    # cria pipeline e busca de melhor modelo
    search = create_pipeline_and_search(
        target,
        lags,
        window_size
    )

    # fit do modelo
    search.fit(X_train, y_train)

    print("\n--------- Importância de features ---------\n")

    # melhor modelo
    best_pipeline = search.best_estimator_
    best_regressor_chain = best_pipeline.named_steps["model"]

    # extraindo o transformador de engenharia de features
    feature_engineering = best_pipeline.named_steps["feature_engineering"]

    # transformando X para obter as features geradas
    X_transformed = feature_engineering.transform(X_train)
    feature_names = X_transformed.columns

    # inicializando a matriz para armazenar as importâncias das features
    num_original_features = X_transformed.shape[1]
    feature_importances = np.zeros(num_original_features)

    # extraindo a importância de features de cada regressor na cadeia
    for estimator in best_regressor_chain.estimators_:
        importances = estimator.feature_importances_[:num_original_features]
        feature_importances[:len(importances)] += importances

    # normalizando as importâncias das features
    feature_importances /= len(best_regressor_chain.estimators_)

    # DataFrame para as importâncias das features
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    importance_csv_path = "./streamlit/importance_df.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    print(importance_df)

    print("\n--------- Validação c/ conjunto de dados de teste ---------\n")

    # extraindo as features e o target para teste
    X_test = test.copy()

    y_test = target_transform(
        test,
        target,
        horizon
    )

    X_test = X_test.iloc[:len(y_test)]  # alinhando X_test e y_test

    # previsões no conjunto de teste
    y_pred = search.predict(X_test)

    # avaliação da performance
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    mape = mean_absolute_percentage_error(y_test, y_pred, multioutput="raw_values")  # noqa 401

    models_performance = pd.DataFrame({
        "Previsão": [f"D+{i}" for i in range(1, horizon+1)],
        "Mean Absolute Error": mae,
        "Mean Absolute Percentage Error": mape
    })

    print(models_performance)

    print(f"\n--------- Previsão de {horizon} dias ---------\n")

    # conjundo de dados para previsão
    X_pred = eia366[-horizon:]

    # feature engineer
    feature_engineer = search.best_estimator_.named_steps["feature_engineering"]  # noqa
    X_pred_transformed = feature_engineer.transform(X_pred)[-1:]

    # normalização
    scaler = search.best_estimator_.named_steps["scaler"]
    scaled_data = scaler.transform(X_pred_transformed)

    # previsões
    model = search.best_estimator_.named_steps["model"]
    predictions = model.predict(scaled_data)

    # previsões
    preds_df = pd.DataFrame(predictions.T)
    preds_df.columns = ["Preço (US$)"]
    last_date = eia366.index[-1]
    new_dates = pd.date_range(start=last_date + BDay(1), periods=15, freq=BDay())  # noqa
    preds_df["Data"] = new_dates
    preds_df = pd.concat([preds_df, models_performance], axis=1)

    preds_csv_path = "./streamlit/preds_df.csv"
    preds_df.to_csv(preds_csv_path, index=False)

    print(f"Histórico de dados (data de ínicio, data de fim): {X_train.index[0]}, {X_train.index[-1]}")  # noqa
    print(f"Histórico de dados (treino): {len(X_train)} dias")
    print(f"Histórico de dados (teste): {len(X_test)} dias")
    print(f"Horizonte de previsão: {horizon} dias")
    print(preds_df)


if __name__ == "__main__":
    main()

```

Crie o arquivo de avariáveis de ambiente para pode usar os módulo do `src/`:

```bash
echo > .env
```

Coloque o conteúdo:

```
PYTHONPATH=.
```