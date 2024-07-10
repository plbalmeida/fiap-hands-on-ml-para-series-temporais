# Testes

A seguir serão criados os arquivos de teste dos módulos do diretório `src/`:

Criar o diretório:

```bash
mkdir tests
echo > tests/feature_engineer_test.py
```

Código do de teste para `src/feature_engineer.py`:

```python
# tests/feature_engineer_test.py

import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
import pandas as pd  # noqa
from src.feature_engineer import FeatureEngineer  # noqa


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        # cria um DataFrame de exemplo para os testes
        data = {
            "date": pd.date_range(start="2022-01-01", periods=10, freq="D"),
            "target": range(10)
        }
        self.df = pd.DataFrame(data)
        self.df.set_index("date", inplace=True)

    def test_lags(self):
        fe = FeatureEngineer(target="target", lags=3, window_size=[2, 3])
        transformed_df = fe.fit_transform(self.df)

        # imprime os valores das colunas de defasagem para inspeção
        print(transformed_df[["lag_1", "lag_2", "lag_3"]])

        # verifica se as colunas de defasagem foram criadas
        self.assertIn("lag_1", transformed_df.columns)
        self.assertIn("lag_2", transformed_df.columns)
        self.assertIn("lag_3", transformed_df.columns)

        # verifica se os valores estão corretos após preenchimento de NaN com 0
        self.assertEqual(transformed_df["lag_1"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_1"].iloc[1], 1)
        self.assertEqual(transformed_df["lag_1"].iloc[2], 2)

        self.assertEqual(transformed_df["lag_2"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_2"].iloc[1], 0)
        self.assertEqual(transformed_df["lag_2"].iloc[2], 1)
        self.assertEqual(transformed_df["lag_2"].iloc[3], 2)

        self.assertEqual(transformed_df["lag_3"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[1], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[2], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[3], 1)
        self.assertEqual(transformed_df["lag_3"].iloc[4], 2)

    def test_rolling_features(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se as colunas de médias móveis foram criadas
        self.assertIn("rolling_mean_2", transformed_df.columns)
        self.assertIn("rolling_std_2", transformed_df.columns)
        self.assertIn("ewm_mean_2", transformed_df.columns)
        self.assertIn("ewm_std_2", transformed_df.columns)

    def test_diff(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se a coluna de diferença foi criada
        self.assertIn("diff", transformed_df.columns)
        # verifica se os valores estão corretos
        self.assertEqual(transformed_df["diff"].iloc[1], 1)

    def test_date_features(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se as colunas de data foram criadas
        self.assertIn("year", transformed_df.columns)
        self.assertIn("quarter", transformed_df.columns)
        self.assertIn("month", transformed_df.columns)
        self.assertIn("day", transformed_df.columns)
        self.assertIn("day_of_week", transformed_df.columns)

    def test_no_target_column(self):
        fe = FeatureEngineer(target="target", lags=1, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se a coluna alvo foi removida
        self.assertNotIn("target", transformed_df.columns)

    def test_fill_na(self):
        fe = FeatureEngineer(target="target", lags=3, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se os valores NaN foram preenchidos com 0
        self.assertFalse(transformed_df.isna().any().any())


if __name__ == "__main__":
    unittest.main()
```

Para o script de teste do `src/model_train.py`

```bash
echo > tests/model_train_test.py
```

Código do `model_train_test.py`:

```python
# tests/model_train_test.py

import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit  # noqa
from src.model_train import create_pipeline_and_search  # noqa


class TestPipelineAndSearch(unittest.TestCase):

    def setUp(self):
        # parâmetros para a função
        self.target = "target"
        self.lags = 3
        self.window_size = [2, 3]

    def test_create_pipeline_and_search(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # verifica se o objeto retornado é uma instância de HalvingGridSearchCV
        self.assertIsInstance(search, HalvingGridSearchCV)

        # verifica se o pipeline está configurado corretamente
        self.assertIn("feature_engineering", search.estimator.named_steps)
        self.assertIn("scaler", search.estimator.named_steps)
        self.assertIn("model", search.estimator.named_steps)

        # verifica se os parâmetros do grid de busca estão corretos
        param_grid = search.param_grid
        self.assertIn("model__base_estimator__n_estimators", param_grid)
        self.assertIn("model__base_estimator__learning_rate", param_grid)
        self.assertIn("model__base_estimator__max_depth", param_grid)
        self.assertIn("model__base_estimator__min_samples_split", param_grid)
        self.assertIn("model__base_estimator__min_samples_leaf", param_grid)

        # verifica os valores dos parâmetros do grid
        self.assertEqual(param_grid["model__base_estimator__n_estimators"], [100, 200, 300])  # noqa
        self.assertEqual(param_grid["model__base_estimator__learning_rate"], [0.01, 0.05, 0.1])  # noqa
        self.assertEqual(param_grid["model__base_estimator__max_depth"], [3, 5, 8])  # noqa
        self.assertEqual(param_grid["model__base_estimator__min_samples_split"], [2, 5, 10])  # noqa
        self.assertEqual(param_grid["model__base_estimator__min_samples_leaf"], [1, 2, 4])  # noqa

    def test_pipeline_structure(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # verifica se a pipeline tem os passos corretos
        steps = search.estimator.steps
        self.assertEqual(steps[0][0], "feature_engineering")
        self.assertEqual(steps[1][0], "scaler")
        self.assertEqual(steps[2][0], "model")

    def test_time_series_split(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # Verifica se o cross-validator é um TimeSeriesSplit
        self.assertIsInstance(search.cv, TimeSeriesSplit)
        self.assertEqual(search.cv.n_splits, 6)


if __name__ == "__main__":
    unittest.main()
```

Para o script de teste do `src/utils.py`

```bash
echo > tests/utils_test.py
```

Código do `utils_test.py`:

```python
# tests/utils_test.py

import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
import pandas as pd  # noqa
from src.utils import target_transform  # noqa


class TestTargetTransform(unittest.TestCase):

    def setUp(self):
        # cria um DataFrame de exemplo para os testes
        self.train = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.target = 'target'
        self.horizon = 3

    def test_target_transform_columns(self):
        # transforma o alvo para múltiplos passos à frente
        transformed = target_transform(self.train, self.target, self.horizon)

        # verifica se as colunas estão nomeadas corretamente
        expected_columns = ['target_t1', 'target_t2', 'target_t3']
        self.assertListEqual(list(transformed.columns), expected_columns)

    def test_target_transform_horizon_one(self):
        # transforma o alvo com horizonte de um passo à frente
        horizon_one = 1
        transformed = target_transform(self.train, self.target, horizon_one)

        # verifica se a coluna está nomeada corretamente
        expected_columns = ['target_t1']
        self.assertListEqual(list(transformed.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()

```

Executar os seguinte comandos para checar os testes:

```bash
python -m unittest tests/feature_engineer_test.py
python -m unittest tests/model_train_test.py
python -m unittest tests/utils_test.py
```
