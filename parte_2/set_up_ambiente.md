Para realizar a parte 2 do hands-on, é necessário o seguinte set up:

1) Ter o git instalado no sua máquina local

- [https://git-scm.com/](https://git-scm.com/)

- [https://www.alura.com.br/artigos/o-que-e-git-github#:~:text=Abra%20o%20Terminal%20e%20digite,instru%C3%A7%C3%B5es%20para%20instalar%20o%20Git.](https://www.alura.com.br/artigos/o-que-e-git-github#:~:text=Abra%20o%20Terminal%20e%20digite,instru%C3%A7%C3%B5es%20para%20instalar%20o%20Git.)

2) Ter conta no GitHub

- [https://git-scm.com/](https://git-scm.com/)

- [https://docs.github.com/pt/get-started/start-your-journey/creating-an-account-on-github](https://docs.github.com/pt/get-started/start-your-journey/creating-an-account-on-github)

3) Ter o Docker Desktop instalado

- [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

4) Ativar o ambiente virtual (o mesmo foi criado na parte 1)

```bash
source venv/bin/activate
```

5) Atualizar o arquivo de dependências

Atualizar o `requirements.txt` com as bibliotecas que iremos usar versionadas.

```txt
altair==5.3.0
ipeadatapy==0.1.9
matplotlib==3.9.0
numpy==1.26.0
pandas==2.2.2
plotly==5.22.0
python-dotenv==1.0.1
scikit-learn==1.5.1
streamlit==1.34.0
```

Instalar as bibliotecas no ambiente virtual criado:

```bash
pip install -r requirements.txt  
```

6) Criar repo no GitHub

Iniciar o repositório localmente:

```bash
git init
```

dicione alguns arquivos ao seu repositório e faça um commit inicial:

```bash
echo "<NOME-DO-PROJETO>" >> README.md
git add README.md
git commit -m "First commit"
```

Criar arquivo `.gitignore` para não subir para o GitHub arquivos indesejáveis:

```bash 
echo .gitignore
```

Coloque o conteúdo nesse arquivo:

```
# ignora o diretório de ambiente virtual
venv/

# ignora todos os diretórios de __pycache__
**/__pycache__/

# ignora todos os arquivos .csv
*.csv
```

Para criar um repositório remoto no GitHub (supondo que você já tenha o GitHub CLI (gh) instalado):

```sh
gh repo create <NOME-DO-PROJETO> --public
```

Agora, adicione a URL do repositório remoto ao seu repositório local:

```sh
git remote add origin https://github.com/<SEU-USUARIO>/<NOME-DO-PROJETO>.git
```

Finalmente, envie seus commits locais para o repositório remoto:

```sh
git push -u origin master
```

Como boa prática de commits no git, vamos usar o Conventional Commits ([https://www.conventionalcommits.org/en/v1.0.0/](https://www.conventionalcommits.org/en/v1.0.0/)).

Conventional Commits é uma convenção simples para formatar mensagens de commit no Git, que fornece um conjunto de regras que facilitam a criação de um histórico de commits explícito e legível. As mensagens de commit no formato Conventional Commits seguem uma estrutura específica que inclui um tipo, um escopo opcional, e uma descrição.

Estrutura de um commit convencional:

```
<tipo>(<escopo opcional>): <descrição>

[corpo opcional]
[rodapé opcional]
```

Tipos comuns:

- **feat**: Uma nova funcionalidade para o usuário.
- **fix**: Uma correção de bug.
- **docs**: Alterações na documentação.
- **style**: Alterações que não afetam o significado do código (espaços em branco, formatação, etc.).
- **refactor**: Uma alteração no código que não corrige um bug nem adiciona uma funcionalidade.
- **perf**: Uma mudança no código que melhora o desempenho.
- **test**: Adição de testes ausentes ou correção de testes existentes.
- **chore**: Alterações em tarefas de build, ferramentas auxiliares, dependências, etc.

Exemplo de commit

```bash
feat(src/main.py): inclusão de egenharia de features 

As abordagens de engenharia de features incluídas no script  extrai novas features melhorando a performance dos modelos de ML implementados.
```

Benefícios:

1. **Histórico Claro**: Facilita a leitura e compreensão do histórico de commits.
2. **Automação**: Permite a automação de processos, como geração de changelogs e versionamento semântico.
3. **Comunicação**: Melhora a comunicação entre os membros da equipe ao fornecer uma descrição clara das mudanças.