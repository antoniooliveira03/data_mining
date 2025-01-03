import plotly.graph_objects as go
import plotly.express as px

def plot_avg_hr_by_label(data, labels_col):
    """
    Plots average HR values by label for each hour in HR_0 to HR_23, grouped by the specified labels column using Plotly.
    
    Args:
    - data: DataFrame containing HR columns and labels column.
    - labels_col: The column name that contains the labels for grouping.
    
    Returns:
    - An interactive line plot displaying average HR values by label using Plotly.
    """
    
    # Select HR columns (exclude 'ratio' columns)
    hr_columns = [column for column in data.columns if 'HR' in column and 'ratio' not in column]
    
    # Calculate the mean values for each label
    label_means = data[hr_columns + [labels_col]].groupby(labels_col).mean()

    # Create the figure
    fig = go.Figure()

    # Add traces for each label
    for label_idx in label_means.index:
        fig.add_trace(go.Scatter(
            x=hr_columns, 
            y=label_means.loc[label_idx],
            mode='lines',
            name=f"Label {label_idx}",
            line=dict(width=2),
            marker=dict(size=6)
        ))

    # Customize the plot
    fig.update_layout(
        title="Average HR Values by Label",
        xaxis_title="Hours (HR_0 to HR_23)",
        yaxis_title="Average Value",
        xaxis=dict(tickangle=45),
        template="plotly_white",
        legend_title="Labels",
        legend=dict(title='Labels', font=dict(size=10)),
        margin=dict(l=20, r=20, t=40, b=80)  # Adjusting margin for better layout
    )

    # Show the plot
    fig.show()

def plot_boxplot_by_cluster(df, labels_col, y_col):
    """
    Plot a box plot of a given column against cluster labels with different colors.
    
    Parameters:
    - df: DataFrame containing the data
    - labels_col: Column name for cluster labels (clusters)
    - y_col: Column name for the y-axis (the variable to compare by clusters)
    
    Returns:
    - A box plot figure
    """
    fig = px.box(df, x=labels_col, y=y_col, color=labels_col,  # color boxes by cluster
                 labels={labels_col: 'Cluster', y_col: y_col},
                 title=f'Box Plot of {y_col} by Cluster')

    fig.update_traces(marker=dict(size=8))  # Adjust boxplot marker size
    fig.show()

def plot_grouped_bar_chart(df, x_col, y_col, labels_col):
    """
    Create a grouped bar chart comparing two categorical variables.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the x-axis (e.g., clusters or any other categorical column)
    - y_col: Column name for the y-axis (e.g., a categorical feature like payment method)
    - labels_col: Column name for the cluster labels

    Returns:
    - A grouped bar chart showing the counts of the second categorical variable (y_col) within each group on the x-axis (x_col).
    """

    # Group by the x and y categories and count occurrences
    count_data = df.groupby([x_col, y_col]).size().reset_index(name='count')

    # Create a grouped bar chart
    fig = px.bar(count_data,
                 x=x_col,                # X-axis (e.g., Clusters or any categorical)
                 y='count',              # Count of occurrences
                 color=y_col,            # Coloring bars by the y-column values (e.g., payment method)
                 barmode='group',        # Grouped bar chart
                 labels={x_col: 'Cluster', y_col: 'Category', 'count': 'Customer Count'},
                 title=f'{y_col} by {x_col} (Grouped)',
                 height=400)             # Adjust the height as needed

    # Show the plot
    fig.show()