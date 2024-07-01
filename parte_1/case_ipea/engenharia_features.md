# Engenharia de Features

A engenharia de features é o processo de criar novas variáveis relevantes a partir de um conjunto de dados original que possam ser utilizadas por algoritmos de ML para melhorar a sua performance de estimação. Este processo envolve a criação, seleção e modificação de variáveis a partir dos dados disponíveis, visando representar de forma mais eficaz os padrões subjacentes e as relações importantes entre os dados. Na prática, isso pode incluir operações como normalização, criação de novas variáveis a partir de combinações das existentes, extração de estatísticas agregadas e tratamento de valores ausentes. Uma boa engenharia de features é essencial para o sucesso de qualquer projeto de ML, pois impacta diretamente a capacidade do modelo de aprender e fazer previsões precisas.

## Engenharia de Features para Série Temporal

A engenharia de features é um passo crucial na preparação dos dados de séries temporais para modelos de ML. A seguir, é listada algumas das features mais comuns extraídas de séries temporais:

- **Lags (Defasagens)**: Valores anteriores da série temporal são usados como características.

- **Médias Móveis**: Média dos valores anteriores para suavizar a série e capturar tendências.

- **Diferenças**: Diferença entre valores consecutivos para remover tendências e tornar a série estacionária.
- **Componentes Sazonais**: Extração de padrões sazonais que se repetem em intervalos regulares.

- **Transformações Estatísticas**: Desvios padrão, variação e outras medidas estatísticas ao longo do tempo.

- **Features Calendáricas**: Dia da semana, mês, feriados e outras variáveis temporais.

A engenharia de features melhora a capacidade do modelo de capturar padrões temporais complexos e aumenta a precisão das previsões.

## Obtendo novas variáveis com Engenharia de Features

Primeiro vamos checar a tipagem do campo de valor e o index de data, é esperado que o campo de valor seja do tipo `float` e o index do tipo `data`:

```python
eia366[["VALUE (US$)"]].info()

<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 13464 entries, 1986-01-04 to 2024-06-18
Data columns (total 1 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   VALUE (US$)  11194 non-null  float64
dtypes: float64(1)
memory usage: 210.4 KB
```

Vamos selecionar somente a variável de interesse, renomear o campo e o index para padronizar:

```python
eia366 = eia366[["VALUE (US$)"]]
eia366.rename(columns={"VALUE (US$)": "value_usd"}, inplace=True)
eia366.index.name = "date"
eia366 = eia366.dropna()
print(eia366)

           value_usd
date                 
1987-05-20      18.63
1987-05-21      18.45
1987-05-22      18.55
1987-05-25      18.60
1987-05-26      18.63
...               ...
2024-06-12      80.52
2024-06-13      81.44
2024-06-14      81.49
2024-06-17      82.45
2024-06-18      84.79

[11194 rows x 1 columns]
```

Cópia dos dados originais.

```python
df = eia366.copy()
```

A seguir são criadas variáveis de lag de 1 a 7, para relacionar a variável dependente com os últimos 7 dias de preço do barril do petróleo.

```python
# lags de 1 a 7 dias do preço
for lag in range(1, 8):
    df[f"lag_{lag}"] = df["value_usd"].shift(lag)
```

Agora vamos criar uma varável de média móvel dos últimos 7 dias do preço.

> **IMPORTANTE!** A função `shift` é usada para evitar vazamento de dados da variável dependente no conjunto de varíáveis independentes, por isso ela será aplicada na criação das variáveis a seguir.

```python
# média móvel de 7 dias do preço
df["rolling_mean_7"] = df["value_usd"].shift(1).rolling(window=7).mean()
```

Na sequência criaremos a variável de diferença.

```python
# diferenças de preço entre os dias
df["diff"] = df["value_usd"].shift(1).diff()
```

Agora serão obtidas as variáveis de sazonalidade, que são as variáveis de mês e dia da semana.

```python
# componentes sazonais
df["month"] = df.index.month
df["day_of_week"] = df.index.dayofweek
```

Assim como a média móvel, é possível obter outras estatísticas móveis, como o desvio padrão.

```python
# desvio padrão móvel do preço na janela de 7 dias
df["rolling_std_7"] = df["value_usd"].shift(1).rolling(window=7).std()
```

A seguir serão obtidas variáveis de calendário.

```python
# variáveis de calendário
df["day"] = df.index.day
df["quarter"] = df.index.quarter
df["year"] = df.index.year
```

Removemos quaisquer linhas com valores NaN que foram criados ao fazer o shift para obtenção de lags, assim como os registros com valores vazios no data set original.

```python
df = df.dropna()
```

Agora vamos visualizar o data frame com as novas variáveis.

```python
print(df.head())

value_usd  lag_1  lag_2  lag_3  lag_4  lag_5  lag_6  lag_7  \
date                                                                     
2002-04-08      26.97  26.72  26.97  25.39  26.97  26.72  26.97  26.06   
2002-04-09      25.39  26.97  26.72  26.97  25.39  26.97  26.72  26.97   
2002-04-10      25.13  25.39  26.97  26.72  26.97  25.39  26.97  26.72   
2002-04-11      24.22  25.13  25.39  26.97  26.72  26.97  25.39  26.97   
2002-04-12      26.36  24.22  25.13  25.39  26.97  26.72  26.97  25.39   

            rolling_mean_7  diff  month  day_of_week  rolling_std_7  
date                                                                 
2002-04-08       26.542857 -0.25      4            0       0.601712  
2002-04-09       26.672857  0.25      4            1       0.577833  
2002-04-10       26.447143 -1.58      4            2       0.730769  
2002-04-11       26.220000 -0.26      4            3       0.866353  
2002-04-12       25.827143 -0.91      4            4       1.069310  
```