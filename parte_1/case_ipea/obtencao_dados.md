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
cod = "EIA366_PBRENT366"
eia366 = ip.timeseries(cod)
print(eia366.head(5))

                        CODE                   RAW DATE  DAY  MONTH  YEAR  \
DATE                                                                        
1986-01-04  EIA366_PBRENT366  1986-01-04T00:00:00-02:00    4      1  1986   
1986-01-05  EIA366_PBRENT366  1986-01-05T00:00:00-02:00    5      1  1986   
1986-01-06  EIA366_PBRENT366  1986-01-06T00:00:00-02:00    6      1  1986   
1986-01-07  EIA366_PBRENT366  1986-01-07T00:00:00-02:00    7      1  1986   
1986-01-08  EIA366_PBRENT366  1986-01-08T00:00:00-02:00    8      1  1986   

            VALUE (US$)  
DATE                     
1986-01-04          NaN  
1986-01-05          NaN  
1986-01-06          NaN  
1986-01-07          NaN  
1986-01-08          NaN  
```

Para visualizar a série temporal obtida:

```python
import matplotlib.pyplot as plt

# plota os dados
eia366[["VALUE (US$)"]].plot(figsize=(10, 5))
plt.title("Série Temporal - EIA366_PBRENT366")
plt.xlabel("Data")
plt.ylabel("US$")
plt.show()
```
