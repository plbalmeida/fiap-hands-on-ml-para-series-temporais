# Esteira de CI

> **O que é CI?** CI, ou Integração Contínua (Continuous Integration), é uma prática de desenvolvimento de software onde os desenvolvedores frequentemente integraram seu código em um repositório compartilhado, várias vezes ao dia. Cada integração é verificada por meio de automações, como testes unitários e de integração, para detectar erros rapidamente. A principal vantagem do CI é a identificação precoce de defeitos e conflitos de código, o que reduz o tempo e o esforço necessários para corrigir problemas, além de melhorar a qualidade do software. Com CI, equipes de desenvolvimento podem entregar novas funcionalidades e correções de forma mais eficiente e confiável.

Nesse projeto utilizaremos o GitHub Actions para implementar uma esteira de CI.

> **O que é o GitHub Actions?** GitHub Actions é uma funcionalidade do GitHub que permite a automação de fluxos de trabalho de desenvolvimento de software diretamente no repositório. Com GitHub Actions, os desenvolvedores podem definir, criar e gerenciar pipelines de integração contínua (CI) e entrega contínua (CD) usando arquivos de configuração YAML. Esses pipelines podem automatizar tarefas como a compilação do código, execução de testes, implantação de aplicações, e muito mais. As ações são desencadeadas por eventos específicos, como push de código, pull requests ou a criação de issues, permitindo uma personalização completa dos fluxos de trabalho. GitHub Actions integra-se perfeitamente com o ecossistema GitHub, proporcionando uma experiência coesa e simplificada para gerenciar o ciclo de vida do desenvolvimento de software.

Criação do diretório para a esteira de CI com GitHub Actions:

```bash
mkdir .github/workflows
cd .github/workflows
echo > ci.yml 
```

Esteira de CI:

```yml
name: CI

on:
  push:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install flake8

    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

  test:
    needs: lint
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Run tests
      run: |
        python -m unittest discover -s tests

```

Este script faz a configuração para uma Integração Contínua (CI), e define um workflow chamado "CI" que é acionado quando há um push para a branch principal (`main`) do repositório. Abaixo está uma explicação detalhada do script:

### Estrutura do Workflow

- **name: CI**: Nome do workflow.

- **on: push: branches: [ main ]**: O workflow é acionado sempre que há um push na branch `main`.

### Jobs

#### Lint Job

O primeiro job se chama `lint`, que é responsável por verificar o código fonte em busca de erros de sintaxe e problemas de formatação usando `flake8`.

- **runs-on: ubuntu-latest**: Este job será executado em um ambiente Ubuntu mais recente.

- **steps**: Define os passos que serão executados como parte deste job.

  1. **Checkout repository**:
     ```yaml
     - name: Checkout repository
       uses: actions/checkout@v2
     ```
     Faz o checkout do código fonte do repositório.

  2. **Set up Python**:
     ```yaml
     - name: Set up Python
       uses: actions/setup-python@v2
       with:
         python-version: '3.9'
     ```
     Configura o ambiente Python com a versão 3.9.

  3. **Install dependencies**:
     ```yaml
     - name: Install dependencies
       run: |
         pip install --upgrade pip
         pip install flake8
     ```
     Atualiza o `pip` e instala o `flake8`.

  4. **Lint with flake8**:
     ```yaml
     - name: Lint with flake8
       run: |
         # stop the build if there are Python syntax errors or undefined names
         flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
         # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
         flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
     ```
     Executa duas verificações com `flake8`:
     - A primeira verifica se há erros de sintaxe ou nomes indefinidos (`E9`, `F63`, `F7`, `F82`). A seguir uma breve descrição dos tipos de erro:
        - O `E9` é referente a erros de sintaxe gerais: Estes são erros que ocorrem quando o Python não consegue interpretar o código. Por exemplo, parênteses não fechados, erros de indentação, etc;
        - O `F63` é referente a erros relacionados a imports: Estes erros ocorrem quando há problemas com as declarações de importação, como importações circulares;
        - `F7` é referente a erros relacionados ao uso de variáveis: Estes erros ocorrem quando há referências a variáveis que não foram definidas.
        - `F82` se trata de erros relacionados a nomes indefinidos em funções: Estes erros ocorrem quando funções ou métodos fazem referência a variáveis ou funções que não existem no escopo atual.
     - A segunda trata todos os erros como avisos e permite complexidade máxima de 10 e largura máxima de linha de 127 caracteres.

#### Test Job

O segundo job se chama `test`, que é responsável por rodar os testes unitários.

- **needs: lint**: Este job só será executado se o job `lint` for bem-sucedido.

- **runs-on: ubuntu-latest**: Este job também será executado em um ambiente Ubuntu mais recente.

- **steps**: Define os passos que serão executados como parte deste job.

  1. **Checkout repository**:
     ```yaml
     - name: Checkout repository
       uses: actions/checkout@v2
     ```
     Faz o checkout do código fonte do repositório.

  2. **Set up Python**:
     ```yaml
     - name: Set up Python
       uses: actions/setup-python@v2
       with:
         python-version: '3.9'
     ```
     Configura o ambiente Python com a versão 3.9.

  3. **Run tests**:
     ```yaml
     - name: Run tests
       run: |
         python -m unittest discover -s tests
     ```
     Executa os testes unitários usando o módulo `unittest` do Python, procurando testes no diretório `tests`.

Em resumo, este workflow realiza duas tarefas principais: 

- primeiro, ele verifica o código fonte com `flake8` para garantir que não haja erros de sintaxe ou problemas de formatação;

- em seguida, executa testes unitários para verificar a funcionalidade do código. Se o job `lint` falhar, o job `test` não será executado.