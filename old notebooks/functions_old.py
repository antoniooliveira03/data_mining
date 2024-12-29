def plot_evaluation_scores(df, path=None):
    """
    Plots R², Silhouette, and Calinski-Harabasz scores for both KMeans and Hierarchical clustering models 
    (with different linkage methods) in one plot for each score, across different numbers of clusters.
    
    Args:
    - df: DataFrame containing clustering results with columns 
           ['Model', 'n_clusters', 'linkage', 'metric', 'R2', 'Silhouette', 'Calinski-Harabasz'].
    """
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Define the scores to plot
    scores = ['R2', 'Silhouette', 'Calinski-Harabasz']
    
    # Initialize a set for storing unique legend labels
    legend_handles = []
    
    # Iterate through each score
    for idx, score_name in enumerate(scores):
        plt.subplot(3, 1, idx+1)

        # Plot KMeans scores if available
        if 'kmeans' in df['Model'].values:
            kmeans_data = df[df['Model'] == 'kmeans']
            line, = plt.plot(
                kmeans_data['n_clusters'], 
                kmeans_data[score_name], 
                marker='o', 
                label='KMeans', 
                linewidth=2
            )
            if idx == 0:
                legend_handles.append(line)  # Add to legend once
        
        # Plot Hierarchical clustering scores for each linkage method
        if 'hierarchical' in df['Model'].values:
            df['metric'] = df['metric'].fillna('euclidean')
            hierarchical_data = df[df['Model'] == 'hierarchical']
            unique_linkages = hierarchical_data[['linkage', 'metric']].drop_duplicates()
            
            for _, row in unique_linkages.iterrows():
                linkage = row['linkage']
                metric = row['metric']
                
                subset = hierarchical_data[
                    (hierarchical_data['linkage'] == linkage) &
                    (hierarchical_data['metric'] == metric)
                ]
                line, = plt.plot(
                    subset['n_clusters'], 
                    subset[score_name], 
                    marker='o', 
                    label=f"{linkage.capitalize()} (Metric: {metric})", 
                    linewidth=2
                )
                if idx == 0:
                    legend_handles.append(line)  # Add to legend once

        # Customize the plot
        plt.title(f"{score_name} Score", fontsize=16)
        plt.xlabel("Number of Clusters", fontsize=12)
        plt.ylabel(f"{score_name} Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(int(min(df['n_clusters'])), int(max(df['n_clusters'])) + 1))
    
    # Add a single legend for all subplots, located outside the last subplot
    plt.legend(
        handles=legend_handles, 
        title='Clustering Methods', 
        fontsize=10, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=2  # Arrange in 2 columns if there are many methods
    )

    # Adjust layout to prevent overlapping subplots
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend
    plt.show()
    if path != None:
        plt.savefig(f'{path}_scores.png')
