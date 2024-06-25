Para realizar o hands-on, é necessário o seguinte set up.

1) Instalar o python

Ter o python (https://www.python.org/) instalado. A a instalação no Linux para distribuição Ubuntu/Debian é realizado com o seguinte comando:

```bash
sudo apt update
sudo apt install python3
```

Após verificar a instalação, no terminal execute para checar a versão do python instalada:

```bash
python3 --version
```

2) Configuração de Ambiente Virtual

Para criar o ambiente virtual, vamos usar o venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

Isso garante que as dependências dos projetos não conflitem entre si.
