# Considerações Finais

Alguns pontos de melhoria são possíveis na implementação da Parte 2:

1) Criar aplicações containerizadas separadas para `model_traning`, `src` e `streamlit`, de modo que cada aplicação com seu respectivo repositórios tenha suas próprias esteiras de CI/CD;

2) Incrementar mais jobs na esteira CI, por exemplo, checagem de cobertura de testes;

3) Incluir um job de deploy da aplicação em alguma cloud púclica (AWS, Azure ou GCP);

O seguinte repositório possui o código para esse hands-on: [https://github.com/plbalmeida/ml-time-series](https://github.com/plbalmeida/ml-time-series)