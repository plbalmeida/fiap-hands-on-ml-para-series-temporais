# Containerização da Aplicação com Docker

> **O que é o Docker?** Docker é uma plataforma de código aberto que automatiza a implementação de aplicações dentro de contêineres de software. Os contêineres são unidades leves e portáteis que encapsulam uma aplicação e todas as suas dependências, incluindo bibliotecas, configurações e arquivos necessários para a execução, garantindo que o software funcione de maneira consistente em diferentes ambientes. Docker permite a criação, teste e implantação de aplicações rapidamente, facilitando a escalabilidade e a gestão de ambientes de desenvolvimento e produção. Além disso, a utilização de contêineres ajuda a isolar as aplicações, aumentando a segurança e simplificando a manutenção e atualização dos sistemas.

Site do Docker: [https://www.docker.com/](https://www.docker.com/)

Na raíz do repo, criar o arquivo `Dockerfile`

```bash
echo > Dockerfile
```

E coloque as instruções:

```
# imagem base do Python
FROM python:3.9-slim

# define o diretório de trabalho
WORKDIR /app

# copia apenas os diretórios e arquivos necessários
COPY model_training /app/model_training
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY streamlit /app/streamlit

# instala os pacotes necessários
RUN pip install --no-cache-dir -r requirements.txt

# define o PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# expõe a porta que o Streamlit usará
EXPOSE 8501

# comando para rodar o main.py e depois o Streamlit
CMD ["sh", "-c", "python model_training/main.py && streamlit run streamlit/app.py --server.port=8501 --server.address=0.0.0.0"]

```

Ainda na raíz do repositório, criar o `docker-compose.yml`:

```bash
echo > docker-compose.yml
```

Colocar as seguintes instruções nesse arquivo:

```yml

version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./streamlit:/app/streamlit:rw

```

Esse arquivo é um arquivo de configuração para definir e gerenciar serviços de contêineres do Docker. Vamos analisar o que cada parte faz:

1. **version: '3.8'**: Define a versão do Docker Compose que está sendo utilizada. A versão 3.8 é uma versão específica do formato de arquivo do Docker Compose.

2. **services**: Define os serviços que serão executados no contêiner. No caso, há um serviço definido chamado `streamlit`.

3. **streamlit**: Este é o nome do serviço. Aqui são definidas as configurações específicas para o serviço `streamlit`.

    - **build: .**: Indica que o Docker deve construir a imagem do serviço usando o `Dockerfile` presente no diretório atual (`.`).

    - **ports**:
      - `"8501:8501"`: Mapeia a porta 8501 do contêiner para a porta 8501 do host. Isso significa que o serviço Streamlit estará acessível na porta 8501 do seu host (máquina local).

    - **volumes**:
      - `./streamlit:/app/streamlit:rw`: Monta o diretório local `./streamlit` no caminho `/app/streamlit` dentro do contêiner. O sufixo `:rw` indica que o volume está montado em modo leitura-escrita (read-write), permitindo que o contêiner e o host façam alterações nos arquivos dentro deste volume.

Em resumo, esse arquivo configura um serviço Streamlit em um contêiner Docker, expondo a porta 8501 para acesso e montando o diretório local `./streamlit` dentro do contêiner para facilitar o desenvolvimento e a persistência de dados.