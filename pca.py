from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import plotly.express as px

# Make wide layout
st.set_page_config(layout="wide")

st.title("PCA")


st.markdown("### 3D PCA")
col1, col2 = st.columns(2)
table_pca3d = col1.empty()
fig_pca3d = col2.empty()

st.markdown("### 2D PCA")
col1, col2 = st.columns(2)
table_pca2d = col1.empty()
fig_pca2d = col2.empty()


# Generate Random Groups of 512-Dimensional Embeddings
embeddings = np.random.rand(2, 512)
embeddings2 = np.random.rand(2, 512)
embeddings3 = np.random.rand(2, 512)

# Concatenate Embeddings
embeddings = np.concatenate([embeddings, embeddings2, embeddings3], axis=0)

print(embeddings)

# Do 3D PCA
pca = PCA(n_components=3)
pca.fit(embeddings)
embeddings_pca = pca.transform(embeddings)

# Write into table
table_pca3d.write(embeddings_pca)

# Show in Plotly 3D Scatter Plot with different colors for each group

fig = px.scatter_3d(
    embeddings_pca,
    x=0,
    y=1,
    z=2,
    opacity=0.7,
    color_discrete_sequence=["red", "red", "blue", "blue", "green", "green"],
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig_pca3d.plotly_chart(fig, use_container_width=True)

# Do 2d PCA
pca = PCA(n_components=2)
pca.fit(embeddings)
embeddings_pca = pca.transform(embeddings)

# Write into table
table_pca2d.write(embeddings_pca)

# Same in 2d
fig = px.scatter(
    embeddings_pca,
    x=0,
    y=1,
    opacity=0.7,
    color=["ID1", "ID1", "ID3", "ID3", "ID2", "ID2"],
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig.update_traces(marker=dict(size=6))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# Grid on
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

fig_pca2d.plotly_chart(fig, use_container_width=True)
