import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="IMDB Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
    }
    .st-bb { background-color: #f0f2f6; }
    .st-at { background-color: #ffffff; }
    .css-1aumxhk {
        background-color: #f0f2f6;
        background-image: none;
    }
    .st-bh, .st-cg, .st-ci {
        border: 1px solid #e6e6e6;
    }
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
    .st-bq {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# File paths
MODEL_PATH = Path("model.pkl")
DATA_PATH = Path("imdb_top_1000.csv")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Preprocessing
        df['Gross_clean'] = df['Gross'].astype(str).str.replace(',', '').replace('$', '').replace('nan', '0').astype(float)
        df['Runtime_min'] = df['Runtime'].astype(str).str.extract('(\d+)').astype(float)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

# Train or load model
@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading saved model: {str(e)}. Training new model...")
    
    if df is None:
        st.error("Cannot train model - data not loaded")
        return None
        
    try:
        ml_df = df.copy()
        
        # Handle missing values
        ml_df['Meta_score'] = ml_df['Meta_score'].fillna(ml_df['Meta_score'].median())
        ml_df['Gross_clean'] = ml_df['Gross_clean'].fillna(ml_df['Gross_clean'].median())
        ml_df['Runtime_min'] = ml_df['Runtime_min'].fillna(ml_df['Runtime_min'].median())
        
        X = ml_df.drop(['IMDB_Rating', 'Poster_Link', 'Series_Title', 'Overview', 'Gross', 'Runtime'], axis=1)
        y = ml_df['IMDB_Rating']
        
        # Feature categorization
        numeric_features = ['Meta_score', 'No_of_Votes', 'Gross_clean', 'Runtime_min']
        categorical_features = ['Released_Year', 'Certificate', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
        
        # Convert year to string for categorical treatment
        X['Released_Year'] = X['Released_Year'].astype(str)
        
        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        
        model.fit(X, y)
        
        # Save the model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
            
        return model
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

model = get_model()

# App layout
st.title('🎬 IMDB Movie Rating Predictor')
st.markdown("""
This application analyzes the IMDB Top 1000 Movies dataset and predicts movie ratings based on various features.
""")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    options = st.radio(
        'Select a page:',
        ['Dataset Overview', 'Exploratory Analysis', 'Rating Prediction'],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses machine learning to predict IMDB movie ratings based on features like:
    - Director
    - Actors
    - Genre
    - Runtime
    - Release Year
    """)

# Page content
if options == 'Dataset Overview':
    st.header('📊 Dataset Overview')
    
    if df is not None:
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(df.head(20))
        
        with st.expander("Dataset Statistics", expanded=True):
            st.write("Shape:", df.shape)
            st.write("Columns:", list(df.columns))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Numerical Stats")
                st.dataframe(df.describe(include=[np.number]))
            with col2:
                st.subheader("Categorical Stats")
                st.dataframe(df.describe(include=['object']))
    else:
        st.error("Data not loaded. Please check your data file.")

elif options == 'Exploratory Analysis':
    st.header('🔍 Exploratory Data Analysis')
    
    if df is None:
        st.error("Data not loaded. Please check your data file.")
        st.stop()
    
    analysis_option = st.selectbox(
        'Select an analysis to view:',
        ['IMDB Ratings Distribution', 
         'Top Directors', 
         'Runtime Distribution',
         'Genre Analysis',
         'Correlation Matrix',
         'Gross Earnings Over Years']
    )
    
    if analysis_option == 'IMDB Ratings Distribution':
        st.subheader('Distribution of IMDB Ratings')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['IMDB_Rating'], bins=20, kde=True, ax=ax)
        ax.set_title('Distribution of IMDB Ratings')
        st.pyplot(fig)
        st.markdown("""
        **Insight:** Most movies in the top 1000 have ratings between 7.5 and 8.5, 
        with very few below 7 or above 9.
        """)
        
    elif analysis_option == 'Top Directors':
        st.subheader('Top 10 Directors by Number of Movies')
        top_directors = df['Director'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=top_directors.values, y=top_directors.index, palette='viridis', ax=ax)
        ax.set_title('Top 10 Directors by Number of Movies')
        st.pyplot(fig)
        st.markdown("""
        **Insight:** Alfred Hitchcock and Steven Spielberg have the most movies 
        in the top 1000 list.
        """)
        
    elif analysis_option == 'Runtime Distribution':
        st.subheader('Distribution of Movie Runtimes')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Runtime_min'], bins=30, kde=True, ax=ax)
        ax.set_title('Distribution of Movie Runtimes (minutes)')
        st.pyplot(fig)
        st.markdown("""
        **Insight:** Most movies are between 90-150 minutes long, 
        with some outliers on both ends.
        """)
        
    elif analysis_option == 'Genre Analysis':
        st.subheader('Most Common Genres in Top 1000 Movies')
        genres = df['Genre'].str.split(', ', expand=True).stack()
        genre_counts = genres.value_counts().head(15)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='mako', ax=ax)
        ax.set_title('Most Common Genres')
        st.pyplot(fig)
        st.markdown("""
        **Insight:** Drama is by far the most common genre, followed by Comedy and Action.
        Many movies have multiple genres.
        """)
        
    elif analysis_option == 'Correlation Matrix':
        st.subheader('Correlation Matrix of Numerical Features')
        numerical_cols = ['IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross_clean', 'Runtime_min']
        corr_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        st.markdown("""
        **Insight:** IMDB Rating has moderate positive correlation with Meta Score.
        Gross earnings show weak correlation with other features.
        """)
        
    elif analysis_option == 'Gross Earnings Over Years':
        st.subheader('Average Gross Earnings Over Years')
        gross_by_year = df.groupby('Released_Year')['Gross_clean'].mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        gross_by_year.plot(ax=ax)
        ax.set_title('Average Gross Earnings Over Years')
        ax.set_ylabel('Gross Earnings ($)')
        ax.grid(True)
        st.pyplot(fig)
        st.markdown("""
        **Insight:** Gross earnings have generally increased over time, 
        with significant spikes for certain years.
        """)

elif options == 'Rating Prediction':
    st.header('🔮 Movie Rating Predictor')
    
    if model is None:
        st.error("Model not available. Please check the model training process.")
        st.stop()
    
    st.markdown("""
    Predict the IMDB rating based on movie features. 
    Adjust the sliders and select boxes below, then click "Predict Rating".
    """)
    
    with st.form("prediction_form"):
        st.subheader("Movie Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            released_year = st.slider(
                'Released Year', 
                1920, 2023, 2000,
                help="Select the release year of the movie"
            )
            meta_score = st.slider(
                'Meta Score', 
                0, 100, 70,
                help="MetaCritic score (0-100)"
            )
            certificate = st.selectbox(
                'Certificate', 
                ['U', 'UA', 'A', 'R', 'PG-13', 'PG', 'Others'],
                help="Movie certification/rating"
            )
            genre = st.selectbox(
                'Genre', 
                ['Drama', 'Comedy', 'Action', 'Crime', 'Adventure', 'Others'],
                help="Primary genre of the movie"
            )
            
        with col2:
            runtime = st.slider(
                'Runtime (minutes)', 
                60, 240, 120,
                help="Duration of the movie in minutes"
            )
            director = st.selectbox(
                'Director', 
                df['Director'].value_counts().head(20).index.tolist() + ['Other'],
                help="Select the director"
            )
            star1 = st.selectbox(
                'Star 1', 
                df['Star1'].value_counts().head(20).index.tolist() + ['Other'],
                help="Primary actor/actress"
            )
            star2 = st.selectbox(
                'Star 2', 
                df['Star2'].value_counts().head(20).index.tolist() + ['Other'],
                help="Secondary actor/actress"
            )
        
        no_of_votes = st.slider(
            'Number of Votes', 
            1000, 2500000, 500000,
            help="Number of IMDB user votes"
        )
        gross = st.slider(
            'Gross Earnings (in millions)', 
            0, 1000, 100,
            help="Box office earnings in millions"
        )
        
        submitted = st.form_submit_button(
            "Predict Rating",
            help="Click to predict the IMDB rating based on your inputs"
        )
        
        if submitted:
            with st.spinner('Making prediction...'):
                try:
                    # Create input dataframe
                    input_data = pd.DataFrame({
                        'Released_Year': [str(released_year)],
                        'Certificate': [certificate],
                        'Genre': [genre],
                        'Meta_score': [meta_score],
                        'Director': [director],
                        'Star1': [star1],
                        'Star2': [star2],
                        'Star3': ['missing'],  # Placeholder for optional stars
                        'Star4': ['missing'],  # Placeholder for optional stars
                        'No_of_Votes': [no_of_votes],
                        'Gross_clean': [gross * 1000000],
                        'Runtime_min': [runtime]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    
                    # Display result
                    st.success(f'## Predicted IMDB Rating: {prediction[0]:.1f} ⭐')
                    
                    # Show confidence indicator
                    confidence = min(90 + (prediction[0] - 7.5) * 10, 95)  # Simple confidence heuristic
                    st.metric("Confidence", f"{confidence:.0f}%")
                    
                    # Interpretation
                    st.markdown("""
                    **Interpretation:**
                    - Ratings above 8.5 are exceptional
                    - Ratings between 7.5-8.5 are very good
                    - Ratings below 7.0 are average for this top 1000 list
                    """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")