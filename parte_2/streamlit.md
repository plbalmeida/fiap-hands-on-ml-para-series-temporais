# Streamlit

> **O que é o Streamlit?** Streamlit é uma biblioteca de código aberto em python que facilita a criação de aplicações web interativas e personalizadas para ciência de dados e ML. Com Streamlit, os desenvolvedores podem transformar scripts de dados em aplicativos funcionais de maneira rápida e eficiente, sem a necessidade de conhecimento aprofundado em desenvolvimento web. A biblioteca permite a visualização de dados, integração com bibliotecas populares como Pandas, Matplotlib e Plotly, e a criação de widgets interativos com poucas linhas de código, tornando-se uma ferramenta poderosa para criar protótipos e compartilhar análises de dados de forma dinâmica e visualmente atraente.

Site do Streamlit: [https://streamlit.io/](https://streamlit.io/)

Criar o diretório e arquivo:

```bash
mkdir streamlit
echo > streamlit/app.py
```

Colocar o seguinte conteúdo no script:

```python
# streamlit/app.py

import altair as alt  # noqa
import pandas as pd  # noqa
import plotly.graph_objects as go  # noqa
import streamlit as st  # noqa
import os


# paths dos arquivos CSV
base_path = "/app/streamlit"
preds_file_path = os.path.join(base_path, "preds_df.csv")
importance_file_path = os.path.join(base_path, "importance_df.csv")

# CSV de previsões e erro
preds_df = pd.read_csv(preds_file_path)
preds_df["Data"] = pd.to_datetime(preds_df["Data"])

st.title("Previsão do Preço do Petróleo Bruto (IPEA)")

fig = go.Figure()

# adiciona trace de preço
fig.add_trace(
    go.Scatter(
        x=preds_df["Data"],
        y=preds_df["Preço (US$)"],
        mode="lines+markers", name="Preço (US$)"
    )
)

# adiciona trace de erro absoluto
fig.add_trace(
    go.Scatter(
        x=preds_df["Data"],
        y=preds_df["Preço (US$)"] - preds_df["Mean Absolute Error"],
        fill=None,
        mode="lines",
        line_color="lightgrey",
        showlegend=False
    )
)

fig.add_trace(
    go.Scatter(
        x=preds_df["Data"],
        y=preds_df["Preço (US$)"] + preds_df["Mean Absolute Error"],
        fill="tonexty",
        mode="lines",
        line_color="lightgrey",
        name='Erro Absoluto'
    )
)

fig.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço (US$)",
    template="plotly_white"
)

st.plotly_chart(fig)

# exibe o dataframe de previsões
st.write(preds_df)

# CSV de Importância de Features
importance_df = pd.read_csv(importance_file_path)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

st.header("Importância das Features")

# cria um gráfico de barras horizontais com Altair
chart = alt.Chart(importance_df).mark_bar().encode(
    x="Importance",
    y=alt.Y("Feature", sort="-x")
)

st.altair_chart(chart, use_container_width=True)

```

O script irá renderizar um gráfico com as previsões e erro em D+15 do preço do petróleo bruto, uma tabela com as previsões e erro, e uma gráfico de importância de features.