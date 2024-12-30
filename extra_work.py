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

# Create the list of categorical features
categorical_features = [
    'customer_region', 'last_promo', 'payment_method', 
    'is_chain', 'promo_DELIVERY', 'promo_DISCOUNT', 'promo_FREEBIE',
    'pay_CARD', 'pay_CASH'
]

# Create the list of binary features
binary_features = [
    'is_chain', 'promo_DELIVERY', 'promo_DISCOUNT', 'promo_FREEBIE',
    'pay_CARD', 'pay_CASH'
]

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

    cols = st.columns([1, 7])
    with cols[0]:
        st.image('./fotos/company.png')
    with cols[1]:
        st.title("Target Sphere Advisors")

    # Dropdown for team members
    st.subheader("Project Team")
    with st.expander("Click to view team details"):
        team_members = {
            "Data Engineer": ("Ana B. Farinha", "20211514"),
            "Project Manager": ("António Oliveira", "20240526"),
            "Business Analyst": ("Mariana Neto", "20211527"),
            "Data Scientist": ("Salvador Domingues", "20240597"),
        }
        for role, (name, student_id) in team_members.items():
            st.write(f"**{role}**: {name} ({student_id})")

    # App description
    st.subheader("ABCDEats Inc. Project")
    st.write('Consumers today are becoming more selective about where they buy their products and \
              where they spend their money. Consequently, it is essential for companies to better \
             understand their clients and be able to tailor sales and discounts to certain groups \
             of customers.')
    st.write(' ')
    st.write('Knowing this, ABCDEats Inc. approached TargetSphere Advisors about a Customer Segmentation \
             project, whose goal was to segment customers into distinct groups based on shared characteristics \
             and their purchasing behaviours. By identifying these unique segments, the company can create more \
             targeted sales strategies, offer personalized discounts and enhance customer satisfaction and loyalty.')

    st.divider()
    
if selected == "Explore Data":
    st.title("Data Exploration")
    st.divider()
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
    st.subheader("Analyse the distribution of Numerical features") 

    def interactive_hist (dataframe):
        box_hist = st.selectbox('Feature', options=numeric_features)
        color_choice = st.color_picker('Select a plot colour', '#1f77b4')
        bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

        hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
        
        plt.title(f"Histogram of {box_hist}")
        st.pyplot(hist)

        
    interactive_hist (data)

    st.divider()

    st.subheader("Analyse the distribution of Categorical features")

    # Defining and calling the function for an interactive bar chart
    def interactive_bar(dataframe):
        cat_feature = st.selectbox('Select Feature', options=categorical_features)
        color_choice = st.color_picker('Select a plot colour', '#1f77b7')

        # If the selected feature is 'region', reclassify it to make it more readable
        if cat_feature == 'customer_region':  # You can add more conditions for other columns like this
            dataframe[cat_feature] = dataframe[cat_feature].apply(lambda x: f"Region {x}")
        
        # Loop through the list of binary features and apply the map operation
        for feature in binary_features:
            if feature in dataframe.columns:
                dataframe[feature] = dataframe[feature].map({0: 'No', 1: 'Yes'})

        # Create the histogram
        bar_chart = px.histogram(dataframe, x=cat_feature)

        # Update the color of the bars using the color choice
        bar_chart.update_traces(marker=dict(color=color_choice))

        # Add additional customization, like titles and axis labels
        bar_chart.update_layout(
            xaxis_title=cat_feature,
            yaxis_title='Count',
            title=f"Distribution of {cat_feature}",
            template="plotly_dark"  # Optional: apply a theme to the chart
        )

        # Display the chart
        st.plotly_chart(bar_chart)

    interactive_bar(data)

    st.divider()


if selected == "Clustering":
    
    st.title("Cluster Exploration")
    st.divider()

    segment_columns = {"Temporal Data": s.temporal_data,
                       "Expense Data": s.spending_orders,
                       "Cuisine Data": s.cuisine_preferences,
                       "Product Data": s.product_vendor}

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

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Define team member details
    team_members = [
        {"name": "Ana B. Farinha", "role": "Data Engineer", "image": "./fotos/ab.jpeg", 'academic_background': f'Bsc in Data Science @ Nova IMS;\n\n Msc in Data Science & Advance Analytics @ Nova IMS'},
        {"name": "António Oliveira", "role": "Project Manager", "image": "./fotos/ant.jpg", 'academic_background': """
        
        📞 +351 916 013 580 | 📧 tzpoliveira@gmail.com | [LinkedIn](https://www.linkedin.com/in/antonio-oliveira02/)  
         
        - **Education**
          - MSc in Data Science & Advanced Analytics (2024-2026) @ Nova IMS.  
          - BSc in Data Science (2021-2024) @ Nova IMS
          - Erasmus @ University of Mannheim (2024).

        - **Experience**: 
          - **Summer Intern** @ NTT Data (2024)
          - **Football Referee** (2022-present)

        - **Extracurriculars**:  
          - Marketing at Nova Formula Student.  
          - Events at Nova IMS Debate Club.  
          - Volunteering since 2016 (CASA, WebSummit).
        """},
        {"name": "Mariana G. Neto", "role": "Business Analyst", "image": "./fotos/mariana.jpg", "academic_background": 'professional hockey player'},
        {"name": "Salvador Domingues",
            "role": "Data Scientist",
            "image": "./fotos/salvador.jpg",
            "academic_background": """
                📞 +351 919 265 520 | 📧 salvadordomingues@gmail.com | [LinkedIn](https://linkedin.com/in/salvador-domingues)  
                
                - **Education**
                    - MSc in Data Science & Advanced Analytics (2024-2026) @ Nova IMS.  
                    - BSc in Computer Science & Business Management (2020-2024) @ Iscte-IUL.
                    - Erasmus @ University of Granada (2022-2023).  

                - **Experience**: 
                    - **Summer Intern** @ NTT Data (2024).  
                    - **Summer Intern** @ Critical Software (2022).  

                - **Extracurriculars**:  
                    - Federated Basketball Player (2009-2018) @ SL Benfica, Carnide Clube.  
                    - Passionate about swimming and fitness.  
            """},

            ]
    
    # Display team members in columns
    cols = st.columns(len(team_members))
    
    for col, member in zip(cols, team_members):
        with col:
            st.image(member["image"])  
            st.subheader(member["name"]) 
            st.write(member["role"]) 

    st.divider()

    # Now add the second section for background information
    st.subheader("Academic Background of Our Team")
    st.write("")
    st.write("")

    # Create two columns: one for image and name, another for background info
    for member in team_members:
        cols = st.columns([1, 3])  
        
        with cols[0]:  
            st.image(member["image"], width=150)
            st.subheader(member["name"])
            st.write("")

        with cols[1]: 
            st.write(member["academic_background"])
            st.write("")

    st.divider()
    cols = st.columns([1, 7])
    with cols[0]:
        st.image('./fotos/company.png')
    with cols[1]:
        st.subheader("TargetSphere Advisors")

    st.write(' ')
    st.write("The TargetSphere Advisors team consists of a \
             Data-Driven organisation focused on implementing \
             Machine Learning solutions for prediction or clustering purposes. \
             We were founded in 2021, and since then have developed several projects \
             in the Machine Learning area of expertise. TargetSphere Advisors started with \
             three members, and recently expanded its team with a fourth member, as it is believed \
             the project presented by ABCDEats Inc. would require more manpower than initially available. ")
    
