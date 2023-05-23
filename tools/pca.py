from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px


def pca3d(matches):
    """
    Perform PCA on embeddings.
    Args:
        embeddings: np.array of shape (n_embeddings, 512)
    Returns:
        embeddings_pca: np.array of shape (n_embeddings, 3)
    """

    # Get Gallery and Detection Embeddings and stich them together in groups
    embeddings = np.concatenate(
        [[match.embedding_gal, match.embedding_det] for match in matches],
        axis=0,
    )

    # Get Identity Names and stich them together in groups
    identity_names = np.concatenate(
        [[match.name, match.name] for match in matches],
        axis=0,
    )

    # Do 3D PCA
    pca = PCA(n_components=3)
    pca.fit(embeddings)
    embeddings_pca = pca.transform(embeddings)

    fig = px.scatter_3d(
        embeddings_pca,
        x=0,
        y=1,
        z=2,
        opacity=0.7,
        color=identity_names,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


def pca2d(matches):
    """
    Perform PCA on embeddings.
    Args:
        embeddings: np.array of shape (n_embeddings, 512)
    Returns:
        embeddings_pca: np.array of shape (n_embeddings, 3)
    """

    # Get Gallery and Detection Embeddings and stich them together in groups
    embeddings = np.concatenate(
        [[match.embedding_gal, match.embedding_det] for match in matches],
        axis=0,
    )

    # Get Identity Names and stich them together in groups
    identity_names = np.concatenate(
        [[match.name, match.name] for match in matches],
        axis=0,
    )

    # Do 2D PCA
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    embeddings_pca = pca.transform(embeddings)

    fig = px.scatter(
        embeddings_pca,
        x=0,
        y=1,
        opacity=0.7,
        color=identity_names,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig
