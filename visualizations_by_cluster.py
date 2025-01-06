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
    
def plot_customer_region_scatter(merged_df, labels_col):
    """
    Creates a scatter plot of customer regions by cluster labels with varying point size based on the count of customers.
    This version ensures that the labels have a distinct color and no gradient.

    Parameters:
    - merged_df (pd.DataFrame): The dataframe containing the columns 'customer_region', labels_col, and customer count data.
    - labels_col (str): The column name containing cluster labels to use for y-axis and point coloring.

    Returns:
    - A Plotly scatter plot.
    """
    
    # Calculate the count of customers for each unique combination of customer_region and label
    merged_df['customer_count'] = merged_df.groupby(['customer_region', labels_col])['customer_region'].transform('count')
    
    # Ensure that 'labels_col' is treated as categorical
    merged_df[labels_col] = merged_df[labels_col].astype(str)  # Convert labels_col to string type to treat it as categorical
    
    # Create a color map for labels (each label gets a distinct color)
    label_colors = {label: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, label in enumerate(merged_df[labels_col].unique())}

    # Create the scatter plot
    fig = px.scatter(
        merged_df,
        x='customer_region',  # Use 'customer_region' for the x-axis
        y=labels_col,  # Use 'label' (cluster labels) for the y-axis
        color=labels_col,  # Color points based on cluster label
        size='customer_count',  # Vary size based on customer count in region and label combination
        color_discrete_map=label_colors,  # Use the predefined label-to-color map
        labels={'customer_region': 'Customer Region', f'{labels_col}': 'Cluster Label', 'customer_count': 'Customer Count'},
        title='Scatter Plot of Customer Regions by Cluster Label with Customer Count Size',
        category_orders={  # Ensures x-axis is categorical, no gradient scale
            'customer_region': merged_df['customer_region'].unique().tolist(),
            labels_col: merged_df[labels_col].unique().tolist(),
        }
    )
    
    # Update x-axis to ensure equal spacing for categorical data
    fig.update_xaxes(type='category')  # Ensures customer_region categories are equally spaced
    
    # Ensure no gradient in the legend
    fig.update_traces(marker=dict(line=dict(width=0)))  # Optional: Removes lines around the scatter points
    
    # Display the plot
    fig.show()

