Para realizar o hands-on, é necessário o seguinte set up:

1) Ter uma IDE (Integrated Development Environment) instalada

Para instalar o VSCode no Linux com distribuição Ubuntu/Debian, executar:

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code
```

2) Ter o python instalado

Para a instalação no Linux com distribuição Ubuntu/Debian executar no terminal:

```bash
sudo apt update
sudo apt install python3
```

Após verificar a instalação, no terminal execute para checar a versão do python instalada:

```bash
python3 --version
```

3) Configuração de Ambiente Virtual

Para criar o ambiente virtual, vamos usar o venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

Isso garante que as dependências dos projetos não conflitem entre si.