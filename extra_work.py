import streamlit as st
from streamlit_option_menu import option_menu
import plotly_express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation as s
import plots as p

data = pd.read_csv('./data/data_for_website_with_labels_no_outliers.csv',
                   index_col = 'customer_id')

# numeric features 
numeric_features = ['customer_age', 'vendor_count', 'product_count', 'is_chain',
                    'first_order', 'last_order',
                    *['HR_' + str(i) for i in range(24)],
                    *['DOW_' + str(i) for i in range(7)],
                    *[col for col in data.columns if col.startswith('CUI_')]]

# categorical features
categorical_features = [
    'customer_region', 'last_promo', 'payment_method', 
    'is_chain', 'promo_DELIVERY', 'promo_DISCOUNT', 'promo_FREEBIE',
    'pay_CARD', 'pay_CASH'
]

# binary features
binary_features = [
    'is_chain', 'promo_DELIVERY', 'promo_DISCOUNT', 'promo_FREEBIE',
    'pay_CARD', 'pay_CASH'
]

all_categ = categorical_features + binary_features

cuisine_preferences = [column for column in data.columns if column.startswith('CUI_') and column.endswith('_ratio')]
product_vendor = [
    'vendor_count', 'is_chain']



## WEBSITE

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
        cat_feature = st.selectbox('Feature', options=categorical_features)
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
                       "Cuisine Data": cuisine_preferences,
                       "Product Data": product_vendor,
                       "All Segments": list(data.columns)}

    segment = st.selectbox("Select Segment to Analyse", list(segment_columns.keys()))

    # Filter data based on selected segment
    selected_columns = segment_columns[segment]
    filtered_data = data[selected_columns]

    label_columns = ["temp_data_labels", "customer_data_labels", "spending_data_labels", "product_data_labels", "cuisine_data_labels", 'merged_labels']
    selectable_columns = [col for col in selected_columns if col not in label_columns]

    # Display selected segment columns
    with st.expander("View Columns in This Segment"):
        st.write(selected_columns)

    st.write(' ')
    st.subheader('Cluster Profiles')


    if segment == "All Segments":
        features = st.multiselect(
            "Select Features to Include", 
            selected_columns, 
            default=selected_columns[2:4]
        )
    else:
        features = st.multiselect(
            "Select Features to Include",
            selectable_columns,
            default=selectable_columns 
        )

    # Allow user to choose clustering label column
    label_column = st.selectbox("Select Clustering Label Column", [col for col in data.columns if col.endswith("labels")])

    # Plot if features are selected
    if features and label_column:
        st.write(f"Plotting Cluster Profiles for {segment} Segment")
        p.cluster_profiles(df=data[features + [label_column]], 
                         label_columns=[label_column],
                         figsize=(10, 6))
    else:
        st.warning("Please select at least one feature and a label column.")


    st.write(' ')
    st.subheader('Dimensionality Reduction Visualisation')

    technique = st.radio("Select Dimensionality Reduction Technique", options=['UMAP', 't-SNE'])

    if technique == 'UMAP':
        n_neighbors = st.slider("Select Number of Neighbors for UMAP", min_value=2, max_value=100, value=15)

        if segment == "All Segments":
            selectable_columns = list(filter(lambda col: col not in all_categ, selectable_columns))
            p.plot_dim_reduction(data[selectable_columns], technique=technique, 
                            n_neighbors=n_neighbors, targets=data[label_column])
            
        else:
            p.plot_dim_reduction(data[selected_columns], technique=technique, 
                            n_neighbors=n_neighbors, targets=data[label_column])
    else:
        p.plot_dim_reduction(data[selected_columns], technique=technique, 
                            targets=data[label_column])


if selected == "About Us":

    st.title("Meet the Team")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Define team member details
    team_members = [
        {"name": "Ana B. Farinha", "role": "Data Engineer", "image": "./fotos/ab.jpeg", "academic_background": """
        📞 +351 969 496 744 | 📧 anabdfarinha@gmail.com | [LinkedIn](https://www.linkedin.com/in/ana-farinha-pt/)  
        
        - **Education**
            - **MSc** in Data Science & Advanced Analytics (2024-2026) @ Nova IMS.  
            - **BSc** in Data Science (2021-2024) @ Nova IMS.  
            - **Erasmus** @ University of Mannheim (2024).  

        - **Experience**: 
            - **Summer Intern** @ Brighten Consulting (2023).  
         
        - **Extracurriculars**:  
            - Ambassador @ Magma Studios (2024).  
            - Volunteer @ Web Summit (2023, 2024).  
        """
        },
        {"name": "António Oliveira", "role": "Project Manager", "image": "./fotos/ant.jpg", 'academic_background': """
        
        📞 +351 916 013 580 | 📧 tzpoliveira@gmail.com | [LinkedIn](https://www.linkedin.com/in/antonio-oliveira02/)  
         
        - **Education**
          - **MSc** in Data Science & Advanced Analytics (2024-2026) @ Nova IMS.  
          - **BSc** in Data Science (2021-2024) @ Nova IMS
          - **Erasmus** @ University of Mannheim (2024).

        - **Experience**: 
          - **Summer Intern** @ NTT Data (2024)
          - **Football Referee** (2022-present)

        - **Extracurriculars**:  
          - Marketing at Nova Formula Student.  
          - Volunteering since 2016 (CASA, WebSummit).
         
        """},
        {"name": "Mariana G. Neto", "role": "Business Analyst", "image": "./fotos/mariana.jpg", "academic_background": """
        📞 +351 963 248 872 | 📧 mariananeto139@gmail.com | [LinkedIn](https://www.linkedin.com/in/marianagneto)  
        
        - **Education**
            - **MSc** in Data Science & Advanced Analytics (2024-Present) @ Nova Information Management School.  
            - **BSc** in Data Science (2021-2024) @ Nova Information Management School.  
            - **Erasmus** @ Hochschule Neu-Ulm, Germany (March 2024 - July 2024).   

        - **Extracurriculars**:
            - Volunteer @ Banco Alimentar Contra a Fome (2013-Present).  
            - Volunteer @ ADRA Portugal (2015-2018).  
    """
},
        {"name": "Salvador Domingues",
            "role": "Data Scientist",
            "image": "./fotos/salvador.jpg",
            "academic_background": """
                📞 +351 919 265 520 | 📧 salvadordomingues@gmail.com | [LinkedIn](https://linkedin.com/in/salvador-domingues)  
                
                - **Education**
                    - **MSc** in Data Science & Advanced Analytics (2024-2026) @ Nova IMS.  
                    - **BSc** in Computer Science & Business Management (2020-2024) @ Iscte-IUL.
                    - **Erasmus** @ University of Granada (2022-2023).  

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
    
    st.write(' ')
    st.markdown('<h2 style="text-align: center;">Our Vision</h2>', unsafe_allow_html=True)
    st.write(' ')
    st.image('./fotos/vision.jpg')

   
    st.write('At TargetSphere Advisors, our vision for extends beyond this project\'s scope and \
            aims to contribute to the broader field of Data Analytics and Business Intelligence. Through \
            our company, we seek to demonstrate how effective data-driven decision-making can unlock deeper \
            insights and guide businesses towards more informed strategies.')

    st.write('The creation of an interactive website showcasing our Cluster Analysis and Data Visualizations \
            represents our commitment to making complex data accessible and engaging for both technical and non-technical \
            users. We envision this platform evolving into a dynamic tool that not only illustrates the power of data but also \
            fosters a collaborative environment for data exploration. In the future, we hope to:')

    
    st.markdown('''
    - **Expand**:  
    Work with more companies in order to develop more projects that provide users with an even richer experience. By continually adding new data and insights, we aim to offer a comprehensive, up-to-date resource for exploring industry-specific trends and patterns.
    
    - **Improve User Engagement**:  
    Enhance the interactivity of the website with more advanced features, such as personalized data dashboards, real-time data updates, and predictive analytics. Our goal is to create a tool that goes beyond static visualization to offer actionable insights for decision-making.
    
    - **Foster Education and Collaboration**:  
    We envision the website becoming a platform for learning and collaboration, where professionals, students, and enthusiasts alike can explore data, share insights, and learn from each other. By making complex analytical techniques more approachable, we hope to inspire the next generation of data scientists and business analysts.
    ''')

    st.write('Ultimately, our vision is to empower businesses and individuals with the tools they need to make data-driven decisions,\
            all while fostering an environment of innovation, collaboration, and continuous learning.')

    st.write(' ')
    st.markdown('<h2 style="text-align: center;">Our Values</h2>', unsafe_allow_html=True)
    st.write(' ')
    st.image('./fotos/values.png')

    st.write('Our values are the foundation of our approach to Data Management. '
         'They guide how we work, how we collaborate, and how we deliver value to our clients and the broader community.')

    st.markdown('''
    1. **Integrity**:  
    We believe in transparency, honesty, and ethical conduct in all our projects. From data collection to analysis, we ensure that our methodologies are robust, and our findings are accurate and reliable. Our commitment to integrity builds trust with our clients and stakeholders.

    2. **Innovation**:  
    We constantly strive to push the boundaries of what's possible with data. Our team is dedicated to exploring new techniques, tools, and technologies to solve complex problems and create innovative solutions. We embrace a mindset of continuous learning and adaptation.

    3. **Collaboration**:  
    We value teamwork and believe that the best solutions come from collective effort. By fostering a collaborative environment, both within our team and with our clients, we create a space where ideas can be freely shared, and diverse perspectives can drive better outcomes.

    4. **Excellence**:  
    We are committed to delivering the highest quality of work. Our focus on excellence means we rigorously test our assumptions, validate our results, and continuously improve our methods to ensure that our clients receive the best possible insights and recommendations.

    5. **Sustainability**:  
    We recognize the impact that data-driven decisions can have on the environment and society. We aim to develop solutions that are not only effective but also sustainable, with long-term benefits for both our clients and the communities we serve.

    6. **Empowerment**:  
    We believe in empowering others with the knowledge and tools to make data-driven decisions. Through education, transparency, and user-friendly solutions, we enable businesses, professionals, and individuals to harness the full potential of their data.
    ''')

    st.divider()
    st.write(' ')
    st.subheader('Let\'s Connect!')

    st.write('We at TargetSphere Advisors are passionate about transforming data into actionable insights that drive meaningful results. '
            'Our team is dedicated to helping businesses make data-driven decisions that lead to success. Whether you\'re looking to explore new strategies, enhance your data capabilities, or collaborate with like-minded professionals, we would love to hear from you.')


    st.write('**Contact Us**')
    st.write('If you have any questions, need more information, or want to learn how we can help you, feel free to reach out. We believe in building relationships and are always open to discussing how we can work together to achieve your goals.')
    cols2 = st.columns(3) 
    with cols2[0]:  
        st.write("**Email 📩**")
        st.write("geral@targetsphere.com")
    with cols2[1]:  
        st.write("**Phone 📞**")
        st.write("+351 21 345 8765")
    with cols2[2]:  
            st.write("**Address 🏢**")
            st.write("Rua da Data, 251, Lisbon, Portugal")

    st.divider()

    st.write('**Join Us in Shaping the Future of Data**')
    st.write('As we continue to innovate and explore new frontiers in data analytics, we invite you to be part of our journey. Stay connected with us through our updates, events, and exciting new projects. Together, we can unlock the power of data to drive better outcomes for businesses and communities alike.')
