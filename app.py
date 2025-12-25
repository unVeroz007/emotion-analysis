# -*- coding: utf-8 -*-
"""
Streamlit App for Emotion Analysis
Nama: Muhammad Galid Avero
NIM: 2311532008
Mata Kuliah: Praktikum Big Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="Emotion Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

class TextPreprocessor:
    """Custom transformer untuk preprocessing teks"""
    
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        
    def preprocess_text(self, text):
        """Preprocess individual text"""
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove user mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short tokens
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            st.error(f"Error preprocessing text: {e}")
            return ""

@st.cache_resource
def load_models():
    """Load semua model dan preprocessing components"""
    try:
        # Load model
        model = joblib.load('models/best_emotion_model.pkl')
        
        # Load preprocessor
        preprocessor = joblib.load('models/text_preprocessor.pkl')
        
        # Load vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load label mapping
        with open('models/label_mapping.pkl', 'rb') as f:
            label_mapping = joblib.load(f)
            
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return model, preprocessor, vectorizer, label_mapping, metadata
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

def predict_emotion(text, model, preprocessor, vectorizer, label_mapping):
    """Fungsi untuk prediksi emosi dari teks"""
    try:
        # Preprocess text
        cleaned_text = preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        emotion = label_mapping.get(prediction, 'unknown')
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            prob_dict = {label_mapping[i]: float(prob) 
                        for i, prob in enumerate(probabilities)}
        else:
            prob_dict = {emotion: 1.0}
            
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_emotion': emotion,
            'predicted_label': int(prediction),
            'probabilities': prob_dict,
            'confidence': float(prob_dict[emotion]) if prob_dict else None
        }
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def create_wordcloud(text, emotion):
    """Create wordcloud visualization"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
        stopwords=stopwords.words('english'),
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'WordCloud - {emotion.capitalize()}', fontsize=16, fontweight='bold')
    ax.axis('off')
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("<h1 class='main-header'>😊 Emotion Analysis App</h1>", unsafe_allow_html=True)
    st.markdown("**Nama:** Muhammad Galid Avero | **NIM:** 2311532008")
    st.markdown("---")
    
    # Load models
    with st.spinner('Loading models...'):
        model, preprocessor, vectorizer, label_mapping, metadata = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🔮 Single Prediction", "📈 Batch Analysis", "📊 Model Info", "📁 About Dataset"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Emotion Labels:**
    1. 😢 Sadness
    2. 😊 Joy
    3. ❤️ Love
    4. 😠 Anger
    5. 😨 Fear
    6. 😲 Surprise
    """)
    
    # Single Prediction Mode
    if app_mode == "🔮 Single Prediction":
        st.markdown("<h2 class='sub-header'>Single Text Prediction</h2>", unsafe_allow_html=True)
        
        # Input text
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
        
        if input_method == "Text Input":
            text_input = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Type your text here... (e.g., 'I am feeling happy today!')"
            )
            
            sample_texts = [
                "I am feeling absolutely wonderful today!",
                "This situation makes me so frustrated and angry",
                "I'm scared about what might happen tomorrow",
                "I love spending time with my family",
                "Wow, that was completely unexpected!"
            ]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("Sample 1", use_container_width=True):
                    text_input = sample_texts[0]
            with col2:
                if st.button("Sample 2", use_container_width=True):
                    text_input = sample_texts[1]
            with col3:
                if st.button("Sample 3", use_container_width=True):
                    text_input = sample_texts[2]
            with col4:
                if st.button("Sample 4", use_container_width=True):
                    text_input = sample_texts[3]
            with col5:
                if st.button("Sample 5", use_container_width=True):
                    text_input = sample_texts[4]
                    
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                text_input = uploaded_file.read().decode("utf-8")
            else:
                text_input = ""
        
        # Prediction button
        if st.button("Analyze Emotion", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner('Analyzing...'):
                    # Make prediction
                    result = predict_emotion(text_input, model, preprocessor, 
                                           vectorizer, label_mapping)
                    
                    if result:
                        # Display results
                        st.markdown("---")
                        
                        # Prediction result
                        emotion_icons = {
                            'sadness': '😢',
                            'joy': '😊',
                            'love': '❤️',
                            'anger': '😠',
                            'fear': '😨',
                            'surprise': '😲'
                        }
                        
                        icon = emotion_icons.get(result['predicted_emotion'], '😐')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Emotion", 
                                    f"{icon} {result['predicted_emotion'].upper()}")
                        with col2:
                            st.metric("Confidence", 
                                    f"{result['confidence']*100:.1f}%")
                        with col3:
                            st.metric("Label Code", result['predicted_label'])
                        
                        # Probabilities chart
                        st.markdown("<h3 class='sub-header'>Emotion Probabilities</h3>", 
                                  unsafe_allow_html=True)
                        
                        prob_df = pd.DataFrame({
                            'Emotion': list(result['probabilities'].keys()),
                            'Probability': list(result['probabilities'].values())
                        }).sort_values('Probability', ascending=False)
                        
                        # Create two columns for chart and table
                        chart_col, table_col = st.columns([2, 1])
                        
                        with chart_col:
                            fig = px.bar(prob_df, x='Emotion', y='Probability',
                                        color='Probability',
                                        color_continuous_scale='viridis',
                                        title='Probability Distribution')
                            fig.update_layout(xaxis_title="Emotion", 
                                            yaxis_title="Probability")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with table_col:
                            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}),
                                       use_container_width=True)
                        
                        # WordCloud
                        st.markdown("<h3 class='sub-header'>Text Analysis</h3>", 
                                  unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.info(text_input)
                        
                        with col2:
                            st.markdown("**Preprocessed Text:**")
                            st.info(result['cleaned_text'])
                        
                        # Create and display wordcloud
                        if result['cleaned_text'].strip():
                            fig_wc = create_wordcloud(result['cleaned_text'], 
                                                    result['predicted_emotion'])
                            st.pyplot(fig_wc)
                        
            else:
                st.warning("Please enter some text to analyze.")
    
    # Batch Analysis Mode
    elif app_mode == "📈 Batch Analysis":
        st.markdown("<h2 class='sub-header'>Batch Text Analysis</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", 
                                       type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'text' in df.columns:
                st.success(f"File loaded successfully! {len(df)} rows found.")
                
                # Show preview
                with st.expander("Preview Data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Analyze All Texts", type="primary"):
                    # Process all texts
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(df['text']):
                        result = predict_emotion(str(text), model, preprocessor,
                                               vectorizer, label_mapping)
                        if result:
                            predictions.append({
                                'Original Text': text[:100] + "..." if len(str(text)) > 100 else text,
                                'Predicted Emotion': result['predicted_emotion'],
                                'Confidence': f"{result['confidence']*100:.1f}%",
                                'Label': result['predicted_label']
                            })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(predictions)
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.markdown("### Emotion Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        emotion_counts = results_df['Predicted Emotion'].value_counts()
                        fig1 = px.pie(values=emotion_counts.values,
                                     names=emotion_counts.index,
                                     title='Emotion Distribution Pie Chart')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.bar(x=emotion_counts.index,
                                     y=emotion_counts.values,
                                     title='Emotion Count Bar Chart',
                                     labels={'x': 'Emotion', 'y': 'Count'})
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("CSV file must contain a 'text' column")
    
    # Model Info Mode
    elif app_mode == "📊 Model Info":
        st.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
        
        if metadata:
            # Model metrics
            st.markdown("### Model Performance")
            metrics = metadata['best_model_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
            with col2:
                st.metric("Test F1-Score", f"{metrics['test_f1']:.4f}")
            with col3:
                st.metric("Test Precision", f"{metrics['test_precision']:.4f}")
            with col4:
                st.metric("Test Recall", f"{metrics['test_recall']:.4f}")
            
            # Dataset info
            st.markdown("### Dataset Information")
            dataset_info = metadata['dataset_info']
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Total Samples", dataset_info['total_samples'])
            with info_col2:
                st.metric("Training Samples", dataset_info['train_samples'])
            with info_col3:
                st.metric("Test Samples", dataset_info['test_samples'])
            with info_col4:
                st.metric("Number of Classes", dataset_info['n_classes'])
            
            # Model details
            st.markdown("### Model Details")
            st.info(f"**Best Model:** {metadata['best_model']}")
            st.info(f"**Timestamp:** {metadata['timestamp']}")
            
            # Feature information
            st.markdown("### Feature Information")
            preprocess_info = metadata['preprocessing_info']
            st.info(f"**Vectorizer Features:** {preprocess_info['vectorizer_features']}")
            st.info(f"**Preprocessor Type:** {preprocess_info['preprocessor_type']}")
    
    # About Dataset Mode
    elif app_mode == "📁 About Dataset":
        st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
        
        try:
            # Load dataset
            df = pd.read_csv('data/cleaned_emotions.csv')
            
            # Basic info
            st.markdown("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show sample data
            with st.expander("View Dataset Sample"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            with st.expander("Column Information"):
                for col in df.columns:
                    st.write(f"**{col}**: {df[col].dtype}")
            
            # Emotion distribution
            if 'emotion_name' in df.columns:
                st.markdown("### Emotion Distribution")
                
                emotion_counts = df['emotion_name'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x=emotion_counts.index,
                                y=emotion_counts.values,
                                title='Emotion Distribution',
                                labels={'x': 'Emotion', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(values=emotion_counts.values,
                                names=emotion_counts.index,
                                title='Emotion Proportion')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display distribution table
                st.dataframe(
                    emotion_counts.reset_index().rename(
                        columns={'index': 'Emotion', 'emotion_name': 'Count'}
                    ),
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.info("Using sample dataset for demonstration...")
            
            # Create sample data for demonstration
            sample_data = pd.DataFrame({
                'text': [
                    "I am feeling happy today",
                    "This makes me sad",
                    "I love this so much",
                    "I'm angry about this situation"
                ],
                'emotion_name': ['joy', 'sadness', 'love', 'anger']
            })
            st.dataframe(sample_data, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Emotion Analysis App | Praktikum Big Data</p>
        <p>Created by Muhammad Galid Avero (2311532008)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Streamlit App for Emotion Analysis
Nama: Muhammad Galid Avero
NIM: 2311532008
Mata Kuliah: Praktikum Big Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="Emotion Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

class TextPreprocessor:
    """Custom transformer untuk preprocessing teks"""
    
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.use_stemming = use_stemming
        
    def preprocess_text(self, text):
        """Preprocess individual text"""
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Remove user mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short tokens
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
            return ' '.join(tokens)
            
        except Exception as e:
            st.error(f"Error preprocessing text: {e}")
            return ""

@st.cache_resource
def load_models():
    """Load semua model dan preprocessing components"""
    try:
        # Load model
        model = joblib.load('models/best_emotion_model.pkl')
        
        # Load preprocessor
        preprocessor = joblib.load('models/text_preprocessor.pkl')
        
        # Load vectorizer
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Load label mapping
        with open('models/label_mapping.pkl', 'rb') as f:
            label_mapping = joblib.load(f)
            
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return model, preprocessor, vectorizer, label_mapping, metadata
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

def predict_emotion(text, model, preprocessor, vectorizer, label_mapping):
    """Fungsi untuk prediksi emosi dari teks"""
    try:
        # Preprocess text
        cleaned_text = preprocessor.preprocess_text(text)
        
        # Vectorize
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        emotion = label_mapping.get(prediction, 'unknown')
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            prob_dict = {label_mapping[i]: float(prob) 
                        for i, prob in enumerate(probabilities)}
        else:
            prob_dict = {emotion: 1.0}
            
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_emotion': emotion,
            'predicted_label': int(prediction),
            'probabilities': prob_dict,
            'confidence': float(prob_dict[emotion]) if prob_dict else None
        }
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

def create_wordcloud(text, emotion):
    """Create wordcloud visualization"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
        stopwords=stopwords.words('english'),
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'WordCloud - {emotion.capitalize()}', fontsize=16, fontweight='bold')
    ax.axis('off')
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("<h1 class='main-header'>😊 Emotion Analysis App</h1>", unsafe_allow_html=True)
    st.markdown("**Nama:** Muhammad Galid Avero | **NIM:** 2311532008")
    st.markdown("---")
    
    # Load models
    with st.spinner('Loading models...'):
        model, preprocessor, vectorizer, label_mapping, metadata = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check the model files.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🔮 Single Prediction", "📈 Batch Analysis", "📊 Model Info", "📁 About Dataset"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Emotion Labels:**
    1. 😢 Sadness
    2. 😊 Joy
    3. ❤️ Love
    4. 😠 Anger
    5. 😨 Fear
    6. 😲 Surprise
    """)
    
    # Single Prediction Mode
    if app_mode == "🔮 Single Prediction":
        st.markdown("<h2 class='sub-header'>Single Text Prediction</h2>", unsafe_allow_html=True)
        
        # Input text
        input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
        
        if input_method == "Text Input":
            text_input = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Type your text here... (e.g., 'I am feeling happy today!')"
            )
            
            sample_texts = [
                "I am feeling absolutely wonderful today!",
                "This situation makes me so frustrated and angry",
                "I'm scared about what might happen tomorrow",
                "I love spending time with my family",
                "Wow, that was completely unexpected!"
            ]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("Sample 1", use_container_width=True):
                    text_input = sample_texts[0]
            with col2:
                if st.button("Sample 2", use_container_width=True):
                    text_input = sample_texts[1]
            with col3:
                if st.button("Sample 3", use_container_width=True):
                    text_input = sample_texts[2]
            with col4:
                if st.button("Sample 4", use_container_width=True):
                    text_input = sample_texts[3]
            with col5:
                if st.button("Sample 5", use_container_width=True):
                    text_input = sample_texts[4]
                    
        else:
            uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
            if uploaded_file is not None:
                text_input = uploaded_file.read().decode("utf-8")
            else:
                text_input = ""
        
        # Prediction button
        if st.button("Analyze Emotion", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner('Analyzing...'):
                    # Make prediction
                    result = predict_emotion(text_input, model, preprocessor, 
                                           vectorizer, label_mapping)
                    
                    if result:
                        # Display results
                        st.markdown("---")
                        
                        # Prediction result
                        emotion_icons = {
                            'sadness': '😢',
                            'joy': '😊',
                            'love': '❤️',
                            'anger': '😠',
                            'fear': '😨',
                            'surprise': '😲'
                        }
                        
                        icon = emotion_icons.get(result['predicted_emotion'], '😐')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Emotion", 
                                    f"{icon} {result['predicted_emotion'].upper()}")
                        with col2:
                            st.metric("Confidence", 
                                    f"{result['confidence']*100:.1f}%")
                        with col3:
                            st.metric("Label Code", result['predicted_label'])
                        
                        # Probabilities chart
                        st.markdown("<h3 class='sub-header'>Emotion Probabilities</h3>", 
                                  unsafe_allow_html=True)
                        
                        prob_df = pd.DataFrame({
                            'Emotion': list(result['probabilities'].keys()),
                            'Probability': list(result['probabilities'].values())
                        }).sort_values('Probability', ascending=False)
                        
                        # Create two columns for chart and table
                        chart_col, table_col = st.columns([2, 1])
                        
                        with chart_col:
                            fig = px.bar(prob_df, x='Emotion', y='Probability',
                                        color='Probability',
                                        color_continuous_scale='viridis',
                                        title='Probability Distribution')
                            fig.update_layout(xaxis_title="Emotion", 
                                            yaxis_title="Probability")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with table_col:
                            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}),
                                       use_container_width=True)
                        
                        # WordCloud
                        st.markdown("<h3 class='sub-header'>Text Analysis</h3>", 
                                  unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.info(text_input)
                        
                        with col2:
                            st.markdown("**Preprocessed Text:**")
                            st.info(result['cleaned_text'])
                        
                        # Create and display wordcloud
                        if result['cleaned_text'].strip():
                            fig_wc = create_wordcloud(result['cleaned_text'], 
                                                    result['predicted_emotion'])
                            st.pyplot(fig_wc)
                        
            else:
                st.warning("Please enter some text to analyze.")
    
    # Batch Analysis Mode
    elif app_mode == "📈 Batch Analysis":
        st.markdown("<h2 class='sub-header'>Batch Text Analysis</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", 
                                       type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if 'text' in df.columns:
                st.success(f"File loaded successfully! {len(df)} rows found.")
                
                # Show preview
                with st.expander("Preview Data"):
                    st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Analyze All Texts", type="primary"):
                    # Process all texts
                    predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(df['text']):
                        result = predict_emotion(str(text), model, preprocessor,
                                               vectorizer, label_mapping)
                        if result:
                            predictions.append({
                                'Original Text': text[:100] + "..." if len(str(text)) > 100 else text,
                                'Predicted Emotion': result['predicted_emotion'],
                                'Confidence': f"{result['confidence']*100:.1f}%",
                                'Label': result['predicted_label']
                            })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(predictions)
                    
                    # Display results
                    st.markdown("### Analysis Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="emotion_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    st.markdown("### Emotion Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        emotion_counts = results_df['Predicted Emotion'].value_counts()
                        fig1 = px.pie(values=emotion_counts.values,
                                     names=emotion_counts.index,
                                     title='Emotion Distribution Pie Chart')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.bar(x=emotion_counts.index,
                                     y=emotion_counts.values,
                                     title='Emotion Count Bar Chart',
                                     labels={'x': 'Emotion', 'y': 'Count'})
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("CSV file must contain a 'text' column")
    
    # Model Info Mode
    elif app_mode == "📊 Model Info":
        st.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
        
        if metadata:
            # Model metrics
            st.markdown("### Model Performance")
            metrics = metadata['best_model_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
            with col2:
                st.metric("Test F1-Score", f"{metrics['test_f1']:.4f}")
            with col3:
                st.metric("Test Precision", f"{metrics['test_precision']:.4f}")
            with col4:
                st.metric("Test Recall", f"{metrics['test_recall']:.4f}")
            
            # Dataset info
            st.markdown("### Dataset Information")
            dataset_info = metadata['dataset_info']
            
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Total Samples", dataset_info['total_samples'])
            with info_col2:
                st.metric("Training Samples", dataset_info['train_samples'])
            with info_col3:
                st.metric("Test Samples", dataset_info['test_samples'])
            with info_col4:
                st.metric("Number of Classes", dataset_info['n_classes'])
            
            # Model details
            st.markdown("### Model Details")
            st.info(f"**Best Model:** {metadata['best_model']}")
            st.info(f"**Timestamp:** {metadata['timestamp']}")
            
            # Feature information
            st.markdown("### Feature Information")
            preprocess_info = metadata['preprocessing_info']
            st.info(f"**Vectorizer Features:** {preprocess_info['vectorizer_features']}")
            st.info(f"**Preprocessor Type:** {preprocess_info['preprocessor_type']}")
    
    # About Dataset Mode
    elif app_mode == "📁 About Dataset":
        st.markdown("<h2 class='sub-header'>Dataset Information</h2>", unsafe_allow_html=True)
        
        try:
            # Load dataset
            df = pd.read_csv('data/cleaned_emotions.csv')
            
            # Basic info
            st.markdown("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show sample data
            with st.expander("View Dataset Sample"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            with st.expander("Column Information"):
                for col in df.columns:
                    st.write(f"**{col}**: {df[col].dtype}")
            
            # Emotion distribution
            if 'emotion_name' in df.columns:
                st.markdown("### Emotion Distribution")
                
                emotion_counts = df['emotion_name'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x=emotion_counts.index,
                                y=emotion_counts.values,
                                title='Emotion Distribution',
                                labels={'x': 'Emotion', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(values=emotion_counts.values,
                                names=emotion_counts.index,
                                title='Emotion Proportion')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display distribution table
                st.dataframe(
                    emotion_counts.reset_index().rename(
                        columns={'index': 'Emotion', 'emotion_name': 'Count'}
                    ),
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.info("Using sample dataset for demonstration...")
            
            # Create sample data for demonstration
            sample_data = pd.DataFrame({
                'text': [
                    "I am feeling happy today",
                    "This makes me sad",
                    "I love this so much",
                    "I'm angry about this situation"
                ],
                'emotion_name': ['joy', 'sadness', 'love', 'anger']
            })
            st.dataframe(sample_data, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Emotion Analysis App | Praktikum Big Data</p>
        <p>Created by Muhammad Galid Avero (2311532008)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()