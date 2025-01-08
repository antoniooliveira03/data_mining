import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import sklearn.manifold
import numpy as np

# Define the function
def cluster_profiles(df, label_columns, figsize, cmap="tab10", compare_titles=None):
    """
    Pass df with labels columns of one or multiple clustering labels. 
    Then specify these label columns to perform the cluster profile according to them.
    """
    if compare_titles is None:
        compare_titles = [""] * len(label_columns)
        
    fig, axes = plt.subplots(nrows=len(label_columns), 
                            ncols=1,
                            figsize=figsize, 
                            constrained_layout=True,
                            squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compare_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i != label]
        dfax = df.drop(drop_cols, axis=1)
        
        # Getting the cluster centroids
        centroids = dfax.groupby(by=label, as_index=False).mean()
        
        # Convert label column to string for plotting
        centroids[label] = centroids[label].astype(str)
        
        # Setting Data
        pd.plotting.parallel_coordinates(
            centroids, 
            label, 
            color=sns.color_palette(cmap),
            ax=ax[0]
        )

        # Remove vertical gridlines
        ax[0].grid(False)
        
        # Hide vertical lines by overlaying a white box
        for spine in ["top", "right", "bottom", "left"]:
            ax[0].spines[spine].set_visible(False)
        
        # Customize layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(text=titl, xy=(0.95, 1.1), xycoords='axes fraction', fontsize=13, fontweight='heavy') 
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), 
                            rotation=40,
                            ha='right')
        
        ax[0].legend(handles, cluster_labels,
                    loc='center left', bbox_to_anchor=(1, 0.5), title=label)
        
    plt.suptitle("Cluster Simple Profiling", fontsize=23)
    st.pyplot(fig)


def plot_dim_reduction(data, technique='UMAP', n_neighbors=15, targets=None, figsize=(10, 7)):
    """
    Plots a 2D representation of high-dimensional data using UMAP or t-SNE.

    Parameters:
    - data (array-like or DataFrame): High-dimensional data to reduce.
    - technique (str): Dimensionality reduction technique name ('UMAP' or 't-SNE').
    - n_neighbors (int): Number of neighbors for UMAP (ignored for t-SNE).
    - targets (array-like): Cluster labels for data points (optional).
    - figsize (tuple): Figure size (default: (10, 7)).
    """
    plt.figure(figsize=figsize)

    # Apply UMAP or t-SNE
    if technique == 'UMAP':
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(data)
    elif technique == 't-SNE':
        reducer = sklearn.manifold.TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(data)
    else:
        raise ValueError(f"Unsupported technique: {technique}. Choose 'UMAP' or 't-SNE'.")

    # Plotting the results
    if targets is not None:
        scatter = plt.scatter(
            embedding[:, 0], 
            embedding[:, 1], 
            c=np.array(targets).astype(int), 
            cmap='tab10'
        )
        
        # Create a legend for clusters
        labels = np.unique(targets)
        handles = []

        for i, label in enumerate(labels):
            color = scatter.cmap(scatter.norm(i))  
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {label}'))

        plt.legend(handles=handles, title='Clusters')

    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5)

    # Title for the plot based on the dimensionality reduction technique
    if technique == 'UMAP':
        plt.title('UMAP Projection')
    elif technique == 't-SNE':
        plt.title('t-SNE Projection')

    # Use Streamlit's st.pyplot to render the Matplotlib plot
    st.pyplot(plt)
    plt.close()  # Close the plot to avoid overlap in future plots
