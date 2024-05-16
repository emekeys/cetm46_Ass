"""
@author: User
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pickled models
with open('C:/Users/MD/Downloads/gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
with open('C:/Users/MD/Downloads/race_ethnicity_encoder.pkl', 'rb') as f:
    race_ethnicity_encoder = pickle.load(f)
with open('C:/Users/MD/Downloads/lunch_encoder.pkl', 'rb') as f:
    lunch_encoder = pickle.load(f)
with open('C:/Users/MD/Downloads/parental_education_encoder.pkl', 'rb') as f:
    parental_education_encoder = pickle.load(f)
with open('C:/Users/MD/Downloads/test_prep_encoder.pkl', 'rb') as f:
    test_prep_encoder = pickle.load(f)
    
with open('C:/Users/MD/Downloads/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('C:/Users/MD/Downloads/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('C:/Users/MD/Downloads/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)



def get_recommendations(cluster_label):
    recommendations = {}
    if cluster_label == 0:
        recommendations['message'] = "Based on the characteristics provided, you belong to the below-average performers group. We recommend offering additional support and resources to help improve your academic performance."
        recommendations['image']='C:/Users/MD/Downloads/The 11 Stages of Getting Your Child to Wipe Himself (Successfully).GIF'
    elif cluster_label == 1:
        recommendations['message'] = "Based on the characteristics provided, you belong to the high achievers group. Consider exploring advanced coursework or enrichment activities to further challenge yourself."
        recommendations['image']='C:/Users/MD/Downloads/Bright Success Needs a Process Quote Mobile Video.GIF'
    elif cluster_label == 2:
        recommendations['message'] = "Based on the characteristics provided, you belong to the average performers group. Providing consistent support and encouragement can help maintain your current academic performance."
        recommendations['image']='C:/Users/MD/Downloads/Animated Greeting Card You Are Amazing GIF - Animated Greeting Card You Are Amazing - Discover & Share GIFs.GIF'
    else:
        recommendations['message'] = "Unable to determine recommendations based on provided characteristics."

    return recommendations

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items(): 
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',['Home','Student Prediction']) #two pages


def predict_cluster(new_data):
    # Ensure new_data is a pandas DataFrame
    if not isinstance(new_data, pd.DataFrame):
        new_data = pd.DataFrame(new_data)

    # Map gender values to integers
    gender_mapping = {'Male': 0, 'Female': 1}
    new_data['gender'] = new_data['gender'].map(gender_mapping)

    # Map race/ethnicity values to integers
    ethnicity_mapping = {'Asian': 0, 'Black': 1, 'Hispanic': 2, 'White': 3, 'Other': 4}
    new_data['race/ethnicity'] = new_data['race/ethnicity'].map(ethnicity_mapping)

    # Map parental level of education values to integers
    education_mapping = {"Some High School": 0, "High School": 1, "Some College": 2,
                         "Associate's Degree": 3, "Bachelor's Degree": 4, "Master's Degree": 5}
    new_data['parental level of education'] = new_data['parental level of education'].map(education_mapping)

    # Map lunch values to integers
    lunch_mapping = {'Free/Reduced': 0, 'Standard': 1}
    new_data['lunch'] = new_data['lunch'].map(lunch_mapping)

    # Map test preparation course values to integers
    prep_course_mapping = {'None': 0, 'Completed': 1}
    new_data['test preparation course'] = new_data['test preparation course'].map(prep_course_mapping)

    # Scale the data
    new_data_scaled = scaler.transform(new_data)

    # Reduce dimensionality using PCA
    new_data_pca = pca.transform(new_data_scaled)

    # Predict the cluster
    cluster = kmeans.predict(new_data_pca)

    # Get recommendations for the predicted cluster
    recommendations = get_recommendations(cluster[0])

    return recommendations



def main():
    data=pd.read_csv('C:/Users/MD/Downloads/StudentsPerformance (1).csv')
    score_columns = ['math score', 'reading score', 'writing score']
    if app_mode=='Home': 
        st.title('STUDENT CLUSTER PREDICTION :') 
        st.image('C:/Users/MD/Downloads/pexels-keira-burton-6147053.jpg')
      
        st.markdown('Dataset used for Prediction:')
        
        
        for col in score_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

# Title of the Streamlit app
        st.title('Students Performance Analysis')

# Display the data used for the analysis
        st.write('Data:')
        st.write(data.head(10))

# Data manipulation
        gender_mean = data.groupby('gender')[score_columns].mean()
        st.subheader('Pie Chart of Gender Distribution')
        gender_counts = data['gender'].value_counts()
        fig10, ax10 = plt.subplots(figsize=(10, 5))
        ax10.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        ax10.axis('equal')
        st.pyplot(fig10)
# Bar Plot: Average Scores by Gender
        st.subheader('BarPlot: Average Scores by Gender')
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        gender_mean.plot(kind='bar', ax=ax1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        st.pyplot(fig1)
        st.subheader('Scatter Plot: Math vs Reading Scores by Gender')
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x='math score', y='reading score', hue='gender', data=data, ax=ax4)
        st.pyplot(fig4)
        
        st.subheader('Box Plot: Scores by Gender')
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='gender', y='math score', data=data, ax=ax3)
        sns.boxplot(x='gender', y='reading score', data=data, ax=ax3)
        sns.boxplot(x='gender', y='writing score', data=data, ax=ax3)
        st.pyplot(fig3)
        st.subheader('Grouped Bar Plot of Scores by Gender and Test')
        fig11, ax11 = plt.subplots(figsize=(10, 5))
        data_melted = data.melt(id_vars=['gender'], value_vars=score_columns, var_name='Test', value_name='Score')
        sns.barplot(x='Test', y='Score', hue='gender', data=data_melted, ax=ax11)
        ax11.set_title('Scores by Gender and Test')
        st.pyplot(fig11)

    elif app_mode=='Student Prediction':
        st.title('Student Performance Recommender')

    # Get user input
        gender = st.radio('Select  your Gender', ['Male', 'Female'])
        ethnicity = st.selectbox('Race/Ethnicity', ['Asian', 'Black', 'Hispanic', 'White', 'Other'])
        education = st.selectbox('Parental Education Level', ["Some High School", "High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"])
        lunch = st.selectbox('Lunch Type', ['Free/Reduced', 'Standard'])
        prep_course = st.selectbox('Test Preparation Course', ['None', 'Completed'])
        math_score = st.number_input("Please enter your math score:")
        reading_score = st.number_input("Please enter your reading score:")
        writing_score = st.number_input("Please enter your writing score:")
        average_score = (math_score + reading_score + writing_score) / 3

        user_input = {
            'gender': [gender],
            'race/ethnicity': [ethnicity],
            'parental level of education': [education],
            'lunch': [lunch],
            'test preparation course': [prep_course],
            'math score': [math_score],
            'reading score': [reading_score],
            'writing score': [writing_score],
            'average_score': [average_score]
    }

        new_data = pd.DataFrame(user_input)

        if st.button("Predict Cluster"):
            recommendations = predict_cluster(new_data)
            st.write(recommendations['message'])
            st.image(recommendations['image'])
            
            st.subheader('Your Scores vs. Overall Distribution')

            # User data for visualization
            user_scores = {
                'Math Score': math_score,
                'Reading Score': reading_score,
                'Writing Score': writing_score
            }

            # Combine user data with overall data for plots
            combined_data = pd.concat([data, new_data], ignore_index=True)

            # Scatter Plot: User vs Overall
            st.subheader('Scatter Plot: Your Scores vs Overall Data')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(x='math score', y='reading score', hue='gender', data=data, ax=ax, alpha=0.5)
            plt.scatter(math_score, reading_score, color='red', label='Your Score')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            st.pyplot(fig)
            
            st.subheader('Box Plot: Your Scores vs Overall Data')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='gender', y='math score', data=combined_data, ax=ax)
            plt.scatter([0], [math_score], color='red', label='Your Math Score')
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='gender', y='reading score', data=combined_data, ax=ax)
            plt.scatter([0], [reading_score], color='red', label='Your Reading Score')
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='gender', y='writing score', data=combined_data, ax=ax)
            plt.scatter([0], [writing_score], color='red', label='Your Writing Score')
            ax.legend()
            st.pyplot(fig)
            # Distribution Plot: User vs Overall
            st.subheader('Distribution Plot: Your Scores vs Overall Data')
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data['math score'], kde=True, color='blue', label='Math Scores', ax=ax)
            plt.axvline(math_score, color='red', linestyle='--', label='Your Math Score')
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data['reading score'], kde=True, color='green', label='Reading Scores', ax=ax)
            plt.axvline(reading_score, color='red', linestyle='--', label='Your Reading Score')
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data['writing score'], kde=True, color='orange', label='Writing Scores', ax=ax)
            plt.axvline(writing_score, color='red', linestyle='--', label='Your Writing Score')
            ax.legend()
            st.pyplot(fig)
            st.subheader('Pie Chart of Gender Distribution')
            gender_counts = combined_data['gender'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
			

                    

if __name__ == "__main__":
    main()
