# Obtenção dos dados do IPEA

No notebook criado `case.ipynb` instale a biblioteca do IPEA para obtenção da série temporal em uma célula do notebook:

```python
!pip install ipeadatapy
```

Depois de instalar, importar a biblioteca em uma nova célula do notebook:

```python
import ipeadatapy as ip
```

Para buscar as séries temporais disponíveis, usar a função list_series:

```python
series = ip.list_series()
print(series)
```

Para obter dados de uma série específica, é necessário saber o código da série. É possível encontrar o código no site do IPEA ou na lista de séries. 

Nesse case vamos usar o preço por barril do petróleo bruto Brent (FOB), o qual tem o código `EIA366_PBRENT366`.

Com o código em mãos, é só usar a função `timeseries` para obter os dados.

```python
codigo_serie = "EIA366_PBRENT366"
df = ip.timeseries(codigo_serie)
print(df)
```

Para visualizar a série temporal obtida:

```python
import matplotlib.pyplot as plt

# plota os dados
df[["VALUE (US$)"]].plot(figsize=(10, 5))
plt.title("Série Temporal - EIA366_PBRENT366")
plt.xlabel("Data")
plt.ylabel("US$")
plt.show()
```
