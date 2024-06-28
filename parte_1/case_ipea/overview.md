# Overview

O Instituto de Pesquisa Econômica Aplicada (IPEA) disponibiliza uma vasta gama de dados econômicos que podem ser utilizados para análise e previsão de séries temporais. Este case aborda a aplicação de ML para previsão de séries temporais utilizando dados do IPEA. 

Nesse case será mostrado na prática como realizar engenharia de features, as práticas recomendadas para treinamento de modelos de regressão supervisionados para previsão, e a avaliação de performance dos modelos de ML com dados de série temporal.

Há disponível uma biblioteca do python para acessar dados de séries temporais do IPEA o qual será usado nesse case: [https://pypi.org/project/ipeadatapy/](https://pypi.org/project/ipeadatapy/)

A série temporal que será utilizada será de preço por barril do petróleo bruto Brent (FOB) com código `EIA366_PBRENT366`, é uma série temporal com dados diários desde 04/01/1986.
