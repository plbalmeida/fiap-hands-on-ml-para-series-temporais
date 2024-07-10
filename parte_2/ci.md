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
name: Python CI

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