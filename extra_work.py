import streamlit as st
from streamlit_option_menu import option_menu
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation as s

# Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('./data/preprocessed_data.csv')

# Create the list of numeric features 
numeric_features = ['customer_age', 'vendor_count', 'product_count', 'is_chain',
                    'first_order', 'last_order',
                    *['HR_' + str(i) for i in range(24)],
                    *['DOW_' + str(i) for i in range(7)],
                    *[col for col in data.columns if col.startswith('CUI_')]]

def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Explore Data", "Clustering", "About Us"],
        icons=["house", "bar-chart-line", 'basket', "person"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected


selected = streamlit_menu()

# Home Page
if selected == "Home":
    st.title("Target Sphere Advisors")

    # Dropdown for team members
    st.subheader("Project Team")
    with st.expander("Click to view team details"):
        team_members = {
            "ROLE1": ("Ana B. Farinha", "XXXX"),
            "ROLE2": ("António Oliveira", "20240526"),
            "ROLE3": ("Mariana Neto", "XXXX"),
            "ROLE4": ("Salvador Domingues", "XXXXX"),
        }
        for role, (name, student_id) in team_members.items():
            st.write(f"**{role}**: {name} ({student_id})")

    # App description
    st.subheader("ABCDEats Inc. Project")
    st.write('Consumers today are becoming more selective about the businesses they support and where they spend their \
             money. This makes it essential for companies to develop a deep understanding of their customer base in order \
             to tailor their services and marketing strategies more effectively.')
    st.divider()
    
if selected == "Explore Data":
    st.title("Model Data and Insights")
    st.subheader("Analyse the pairwise relation between the numerical features")

    # Defining and calling the function for an interactive scatterplot
    def interactive_scater (dataframe):
        x_axis_val = st.selectbox('Select X-Axis Value', options=numeric_features)
        y_axis_val = st.selectbox('Select Y-Axis Value', options=numeric_features)
        col = st.color_picker('Select a plot colour')

        plot  = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)
        plot.update_traces(marker = dict(color=col))
        st.plotly_chart(plot)

    interactive_scater (data)

    st.divider()
    
    # Creating Hist of numerical
    st.subheader("Analyse the histograms of Numerical features") 

    def interactive_hist (dataframe):
        box_hist = st.selectbox('Feature', options=numeric_features)
        color_choice = st.color_picker('Select a plot colour', '#1f77b4')
        bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

        hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
        
        plt.title(f"Histogram of {box_hist}")
        st.pyplot(hist)

        
    interactive_hist (data)

    st.divider()

if selected == "Clustering":
    
    st.title("Cluster Analysis")

    segment_columns = {"Temporal Data": s.temporal_data,
                       "Expense Data": s.spending_orders}

    segment = st.selectbox("Select Segment to Analyse", list(segment_columns.keys()))

    # Filter data based on selected segment
    selected_columns = segment_columns[segment]
    filtered_data = data[selected_columns]

    # Display selected segment columns
    st.write(f"Columns in {segment} Segment")
    with st.expander("View Columns in This Segment"):
        st.write(selected_columns)

    scaled_data = scaled_data = RobustScaler().fit_transform(filtered_data)


    # HIERARCHICAL
    st.subheader("Hierarchical Clustering")
    linkage_method = st.selectbox("Select Linkage Method", ["ward", "single", "complete", "average"])

    #linkage_matrix = linkage(data[numeric_features], method=linkage_method)

    # Plot the dendrogram
    #st.subheader(f"Dendrogram Using {linkage_method.capitalize()} Linkage")
    #fig, ax = plt.subplots(figsize=(10, 6))
    #dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, ax=ax)
    #plt.title(f"Dendrogram ({linkage_method.capitalize()} Linkage)")
    #plt.xlabel("Data Points")
    #plt.ylabel("Distance")
    #st.pyplot(fig)

    # K-MEANS
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    st.subheader("K-Means Clustering")


    # Select number of clusters
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3, step=1)

    # Apply K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_data['Cluster'] = kmeans.fit_predict(scaled_data)

    # Visualize Clusters using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    filtered_data['PCA1'] = pca_result[:, 0]
    filtered_data['PCA2'] = pca_result[:, 1]

    st.subheader("Cluster Visualization")
    fig = px.scatter(
        filtered_data, x='PCA1', y='PCA2',
        color='Cluster',
        title=f"K-Means Clustering with {num_clusters} Clusters",
        color_continuous_scale=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig)

    # Display Cluster Centroids
    st.subheader("Cluster Centroids")
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns)
    st.dataframe(centroids)

    # View Cluster Sizes
    st.subheader("Cluster Sizes")
    cluster_sizes = filtered_data['Cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'Count']
    st.dataframe(cluster_sizes)

    st.subheader("Cluster Means")
    # View Mean values for each cluster
    selected_cluster = st.selectbox("Select Cluster", range(num_clusters))
    cluster_data = filtered_data[filtered_data['Cluster'] == selected_cluster]

    # Calculate the mean values for the selected cluster
    cluster_means = cluster_data[selected_columns].mean()

    # Convert to DataFrame and set the correct column names
    cluster_means_df = cluster_means.to_frame()
    cluster_means_df = cluster_means_df.T

    # Reset index to remove the default index column and hide it
    cluster_means_df.reset_index(drop=True, inplace=True)

    # Split the DataFrame into chunks of 9 columns each
    chunk_size = 9
    for start in range(0, len(cluster_means_df.columns), chunk_size):
        end = start + chunk_size
        chunk = cluster_means_df.iloc[:, start:end]
        
        # Display the means with improved formatting for each chunk
        st.dataframe(chunk, use_container_width=True)


if selected == "About Us":

    st.title("Meet the Team")
    
    # Define team member details
    team_members = [
        {"name": "Ana B. Farinha", "role": "Data Scientist", "image": "./test.png"},
        {"name": "António Oliveira", "role": "Data Engineer", "image": "./test.png"},
        {"name": "Mariana XX Neto", "role": "Business Analyst", "image": "./test.png"},
        {"name": "Salvador Domingues", "role": "Project Manager", "image": "./test.png"},
    ]
    
    # Display team members in columns
    cols = st.columns(len(team_members))
    
    for col, member in zip(cols, team_members):
        with col:
            st.image(member["image"])  
            st.subheader(member["name"]) 
            st.write(member["role"]) 

    st.divider()

    st.markdown("""
            Our team is dedicated to leveraging data science and business intelligence to deliver valuable insights and innovative solutions. 
            Each team member brings a unique set of skills, from data engineering and analysis to project management and business strategy.
            """)
    