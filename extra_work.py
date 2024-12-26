import streamlit as st
from streamlit_option_menu import option_menu
import plotly_express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./data/preprocessed_data.csv')

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
    st.markdown("""
                App Description
                """)
    
if selected == "Explore Data":
    st.title("Model Data and Insights")
    st.subheader("Analyse the pairwise relation between the numerical features")

    
    # Create the list of numeric features 
    numeric_features = ['customer_age', 'vendor_count', 'product_count', 'is_chain',
                        'first_order', 'last_order',
                        *['HR_' + str(i) for i in range(24)],
                        *['DOW_' + str(i) for i in range(7)],
                        *[col for col in data.columns if col.startswith('CUI_')]]
    
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
    
    #Creating Hist of numerical
    st.subheader("Analyse the histograms of Numerical features") 

    def interactive_hist (dataframe):
        box_hist = st.selectbox('Feature', options=numeric_features)
        color_choice = st.color_picker('Select a plot colour', '#1f77b4')
        bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

        hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
        
        plt.title(f"Histogram of {box_hist}")
        st.pyplot(hist)

        
    interactive_hist(data)

    st.divider()

if selected == "Clustering":
    st.title("Clustering Analysis")