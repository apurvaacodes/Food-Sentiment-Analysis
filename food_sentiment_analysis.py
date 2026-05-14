import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Page Configuration
st.set_page_config(
    page_title="Food Sentiment Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #95E1D3;
        --danger-color: #F38181;
        --dark-bg: #2C3E50;
    }
    
    /* Banner styling */
    .banner {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #FF6B6B;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #FF5252;
        box-shadow: 0 4px 12px rgba(255,107,107,0.3);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2C3E50;
    }
    
    /* Sentiment badges */
    .positive-badge {
        background-color: #95E1D3;
        color: #27AE60;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .negative-badge {
        background-color: #F38181;
        color: #C0392B;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Smaller metric text */
    .small-metric {
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("sentiment_model.pkl")
        return vectorizer, model
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {str(e)}")
        st.stop()

vectorizer, model = load_models()

# Session State Management
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

# Sidebar
with st.sidebar:
    st.image("sentiment_image.png")
    st.title("📊 About Project")
    st.info("""
    **Objective:** Predict sentiment (Negative/Positive) of food reviews using Machine Learning.
    
    **Model Type:** Text Classification
    **Algorithm:** Logistic Regression with TF-IDF Vectorization
    """)
    
    st.title("🛠️ Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Libraries:**
        - NumPy
        - Pandas
        - Scikit-Learn
        """)
    with col2:
        st.markdown("""
        **Deployment:**
        - Streamlit
        - Python 3.x
        """)
    
    st.title("📞 Contact")
    st.markdown("📱 9999999999")
    st.markdown("📧 apurvaa.workmail@gmail.com")
    
    st.divider()
    st.markdown("**Version:** 2.0 | **Last Updated:** 2026-05-13")

# Main Banner
st.markdown("""
<div class="banner">
🍔 Food Sentiment Analysis 🍔
</div>
""", unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Review", "📁 Bulk Analysis", "📈 Analytics", "📜 History"])

# ==================== TAB 1: SINGLE REVIEW ====================
with tab1:
    st.header("Predict Single Review")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Review")
        review_text = st.text_area(
            "Type or paste a food review here:",
            value=st.session_state.review_text,
            height=150,
            placeholder="e.g., 'This pizza was absolutely delicious! Great service too.'"
        )
        
        # Update session state with current text input
        st.session_state.review_text = review_text
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_btn = st.button("🔍 Predict Sentiment", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        # Clear button functionality
        if clear_btn:
            st.placeholder = ""
            st.rerun()
    
    with col2:
        st.subheader("📊 Prediction Details")
        if predict_btn and st.session_state.review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                X_test = vectorizer.transform([st.session_state.review_text])
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)
                
                # Store in history
                st.session_state.predictions_history.append({
                    'review': st.session_state.review_text,
                    'sentiment': 'Positive' if pred[0] == 1 else 'Negative',
                    'confidence': max(prob[0]),
                    'timestamp': datetime.now()
                })
                
                # Display Results
                sentiment = "Positive 👍" if pred[0] == 1 else "Negative 👎"
                confidence = max(prob[0])
                
                if pred[0] == 1:
                    st.success(f"### Sentiment: {sentiment}")
                    st.metric("Confidence Score", f"{confidence:.2%}", delta="Positive")
                else:
                    st.error(f"### Sentiment: {sentiment}")
                    st.metric("Confidence Score", f"{confidence:.2%}", delta="Negative")
                
                # Probability Distribution
                st.subheader("Probability Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Negative', 'Positive'],
                        y=[prob[0][0], prob[0][1]],
                        marker=dict(color=['#F38181', '#95E1D3']),
                        text=[f'{prob[0][0]:.2%}', f'{prob[0][1]:.2%}'],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    yaxis_title="Probability",
                    xaxis_title="Sentiment Class"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif predict_btn:
            st.warning("⚠️ Please enter a review first!")

# ==================== TAB 2: BULK ANALYSIS ====================
with tab2:
    st.header("📁 Bulk Review Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Select a CSV or TXT file",
            type=["csv", "txt"],
            help="File should have reviews in the first column"
        )
    
    with col2:
        st.subheader("File Format Guide")
        st.info("""
        **Expected Format:**
        - Single column with review text
        - No header required
        - One review per row
        
        **Example:**
        ```
        Great food and service
        Terrible experience
        Amazing place!
        ```
        """)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None, names=["Review"])
            
            st.subheader(f"📋 Loaded {len(df)} Reviews")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='small-metric'><p style='font-size: 0.85rem;'>Total Reviews</p><p style='font-size: 1.5rem; font-weight: bold;'>{len(df)}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='small-metric'><p style='font-size: 0.85rem;'>File Size</p><p style='font-size: 1.5rem; font-weight: bold;'>{uploaded_file.size / 1024:.2f} KB</p></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='small-metric'><p style='font-size: 0.85rem;'>Status</p><p style='font-size: 1.5rem; font-weight: bold;'>Ready for Analysis</p></div>", unsafe_allow_html=True)
            
            st.divider()
            
            # Preview
            with st.expander("👁️ Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Predict Button
            if st.button("🚀 Run Bulk Prediction", use_container_width=True):
                with st.spinner("Processing reviews..."):
                    progress_bar = st.progress(0)
                    
                    X_test = vectorizer.transform(df['Review'])
                    pred = model.predict(X_test)
                    prob = model.predict_proba(X_test)
                    
                    sentiment = ["Positive" if i == 1 else "Negative" for i in pred]
                    df['Sentiment'] = sentiment
                    df['Confidence'] = np.max(prob, axis=1)
                    df['Prediction_Probability'] = prob.max(axis=1)
                    
                    progress_bar.progress(100)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    positive_count = (df['Sentiment'] == 'Positive').sum()
                    negative_count = (df['Sentiment'] == 'Negative').sum()
                    avg_confidence = df['Confidence'].mean()
                    
                    with col1:
                        st.metric("Positive Reviews", positive_count, f"{positive_count/len(df)*100:.1f}%")
                    with col2:
                        st.metric("Negative Reviews", negative_count, f"{negative_count/len(df)*100:.1f}%")
                    with col3:
st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    st.divider()
                    
                    # Results Table
                    st.subheader("📊 Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download Results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.predictions_history:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            positive_pct = (history_df['sentiment'] == 'Positive').sum() / len(history_df) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col3:
            avg_conf = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        st.divider()
        
        # Sentiment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = history_df['sentiment'].value_counts()
            fig = go.Figure(data=[
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    marker=dict(colors=['#95E1D3', '#F38181'])
                )
            ])
            fig.update_layout(title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                history_df,
                x='sentiment',
                y='confidence',
                color='sentiment',
                color_discrete_map={'Positive': '#95E1D3', 'Negative': '#F38181'}
            )
            fig.update_layout(title="Confidence Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence over time
        st.subheader("Confidence Trend Over Time")
        fig = px.line(
            history_df,
            x='timestamp',
            y='confidence',
            color='sentiment',
            markers=True,
            color_discrete_map={'Positive': '#95E1D3', 'Negative': '#F38181'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 No predictions yet. Make some predictions to see analytics!")

# ==================== TAB 4: HISTORY ====================
with tab4:
    st.header("📜 Prediction History")
    
    if st.session_state.predictions_history:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Sort by timestamp
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.predictions_history = []
                st.rerun()
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment:",
                options=['Positive', 'Negative'],
                default=['Positive', 'Negative']
            )
        with col2:
            min_confidence = st.slider("Minimum Confidence:", 0.0, 1.0, 0.0)
        
        # Apply filters
        filtered_df = history_df[
            (history_df['sentiment'].isin(sentiment_filter)) &
            (history_df['confidence'] >= min_confidence)
        ]
        
        # Display table
        st.dataframe(
            filtered_df[['timestamp', 'review', 'sentiment', 'confidence']],
            use_container_width=True
        )
        
        # Export history
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History as CSV",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.info("📜 No prediction history yet. Start making predictions!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #7F8C8D; margin-top: 30px;'>
    <p>🔐 Food Sentiment Analysis v2.0</p>
    <p>Built with ❤️ using Streamlit | Created By - Apurva Sharma
</div>
""", unsafe_allow_html=True)
