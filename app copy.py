#!/usr/bin/env python3
"""
NBA Lineup Efficiency Predictor - Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging

# Add utils to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
try:
    from ee_utils import DataProcessor
    from ml_utils import BayesianNetworkModel
except ImportError as e:
    st.error(f"Import error: {e}. Make sure utils/ directory exists with ee_utils.py and ml_utils.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="NBA Lineup Efficiency Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-result {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-top: 1rem;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .talent-input {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'model' not in st.session_state:
        st.session_state.model = BayesianNetworkModel()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data_stats' not in st.session_state:
        st.session_state.data_stats = {}
    if 'training_results' not in st.session_state:
        st.session_state.training_results = {}
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

def update_status():
    """Update system status"""
    return {
        'data_loaded': st.session_state.data_loaded,
        'model_trained': st.session_state.model_trained,
        'data_count': len(st.session_state.data_processor.processed_data) if st.session_state.data_loaded else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_accuracy': st.session_state.training_results.get('accuracy', 0) if st.session_state.model_trained else 0
    }

def load_data():
    """Load existing data"""
    try:
        with st.spinner("Loading sample NBA data..."):
            df = st.session_state.data_processor.load_sample_data()
            st.session_state.data_loaded = True
            st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
        return True
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def download_new_data():
    """Download new data from NBA API"""
    try:
        with st.spinner("Downloading new NBA data..."):
            df = st.session_state.data_processor.download_nba_data()
            st.session_state.data_loaded = True
            st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
        return True
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False

def train_model():
    """Train the Bayesian Network model"""
    if not st.session_state.data_loaded:
        return False
    
    try:
        with st.spinner("Training Bayesian Network model..."):
            processed_df, preprocessing_metadata = st.session_state.data_processor.preprocess_data(
                st.session_state.data_processor.processed_data
            )
            
            results = st.session_state.model.train_model(processed_df, preprocessing_metadata)
            st.session_state.model_trained = True
            st.session_state.training_results = results
        return True
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return False

def create_talent_distribution_plot():
    """Create talent distribution visualization"""
    if not st.session_state.data_loaded:
        return None
    
    try:
        df = st.session_state.data_processor.processed_data
        
        # Create talent distribution plot
        talent_cols = ['scoring_talent', 'playmaking_talent', 'rebounding_talent', 'defensive_talent']
        talent_data = []
        
        for col in talent_cols:
            counts = df[col].value_counts()
            for talent, count in counts.items():
                talent_data.append({
                    'Talent Type': col.replace('_talent', '').title(),
                    'Level': talent,
                    'Count': count
                })
        
        talent_df = pd.DataFrame(talent_data)
        
        fig = px.bar(
            talent_df, 
            x='Talent Type', 
            y='Count', 
            color='Level',
            title='Talent Distribution Across Lineups',
            color_discrete_sequence=px.colors.qualitative.Set2,
            barmode='group'
        )
        
        fig.update_layout(
            xaxis_title="Talent Type",
            yaxis_title="Number of Lineups",
            legend_title="Talent Level",
            height=400
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None

def create_efficiency_distribution_plot():
    """Create efficiency distribution visualization"""
    if not st.session_state.data_loaded:
        return None
    
    try:
        df = st.session_state.data_processor.processed_data
        
        # Create efficiency distribution plot
        efficiency_counts = df['efficiency'].value_counts().reset_index()
        efficiency_counts.columns = ['Efficiency', 'Count']
        
        # Define color mapping for efficiency levels
        color_map = {
            'Very Poor': 'red',
            'Poor': 'orange',
            'Good': 'blue',
            'Excellent': 'green'
        }
        
        fig = px.pie(
            efficiency_counts,
            values='Count',
            names='Efficiency',
            title='Lineup Efficiency Distribution',
            color='Efficiency',
            color_discrete_map=color_map
        )
        
        fig.update_layout(height=400)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating efficiency plot: {e}")
        return None

def create_probability_breakdown_plot(prediction_data):
    """Create probability breakdown visualization for predictions"""
    try:
        prob_data = []
        for eff, prob in prediction_data['all_probabilities'].items():
            prob_data.append({
                'Efficiency': eff,
                'Probability': prob * 100
            })
        
        prob_df = pd.DataFrame(prob_data)
        
        # Color mapping
        color_map = {
            'Very Poor': 'red',
            'Poor': 'orange',
            'Good': 'blue',
            'Excellent': 'green'
        }
        
        fig = px.bar(
            prob_df,
            x='Efficiency',
            y='Probability',
            color='Efficiency',
            color_discrete_map=color_map,
            text='Probability',
            title='Prediction Probability Breakdown'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            yaxis_title="Probability (%)",
            showlegend=False,
            height=300
        )
        fig.update_yaxes(range=[0, 100])
        
        return fig
    except Exception as e:
        logger.error(f"Error creating probability plot: {e}")
        return None

def display_data_insights():
    """Display data insights and statistics"""
    if not st.session_state.data_loaded:
        return
    
    st.subheader("📊 Data Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_records = st.session_state.data_stats.get('total_records', 0)
        st.metric("Total Lineups", f"{total_records:,}")
    
    with col2:
        efficiency_dist = st.session_state.data_stats.get('efficiency_distribution', {})
        excellent_pct = (efficiency_dist.get('Excellent', 0) / total_records * 100) if total_records > 0 else 0
        st.metric("Excellent Lineups", f"{excellent_pct:.1f}%")
    
    with col3:
        if st.session_state.model_trained:
            accuracy = st.session_state.training_results.get('accuracy', 0) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Model Status", "Not Trained")
    
    # Display visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_talent = create_talent_distribution_plot()
        if fig_talent:
            st.plotly_chart(fig_talent, use_container_width=True)
    
    with col2:
        fig_efficiency = create_efficiency_distribution_plot()
        if fig_efficiency:
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Display data sample
    st.subheader("Sample Data")
    if st.session_state.data_loaded:
        sample_data = st.session_state.data_processor.processed_data.head(10)
        st.dataframe(sample_data, use_container_width=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🏀 NBA Lineup Efficiency Predictor</h1>
        <p class="lead" style="font-size: 1.2rem; margin-bottom: 0;">Using Bayesian Networks to predict NBA lineup performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Section
    st.header("📈 System Status")
    
    status = update_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if status['data_loaded'] else "🔴"
        st.markdown(f"""
        <div class="status-card">
            <h3>{status_color}</h3>
            <strong>Data Loaded</strong><br>
            <span style="color: {'green' if status['data_loaded'] else 'red'}">
                {'Yes' if status['data_loaded'] else 'No'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_color = "🟢" if status['model_trained'] else "🔴"
        st.markdown(f"""
        <div class="status-card">
            <h3>{status_color}</h3>
            <strong>Model Trained</strong><br>
            <span style="color: {'green' if status['model_trained'] else 'red'}">
                {'Yes' if status['model_trained'] else 'No'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="status-card">
            <h3>📊</h3>
            <strong>Data Points</strong><br>
            {status['data_count']:,}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="status-card">
            <h3>⏰</h3>
            <strong>Last Update</strong><br>
            {status['timestamp']}
        </div>
        """, unsafe_allow_html=True)
    
    # Control Section
    st.header("🎮 Data & Model Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Load Sample Data", use_container_width=True, help="Load sample NBA lineup data"):
            if load_data():
                st.success("✅ Data loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to load data")
    
    with col2:
        if st.button("🔄 Download New Data", use_container_width=True, help="Download fresh data from NBA API"):
            if download_new_data():
                st.success("✅ New data downloaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to download data")
    
    with col3:
        if st.button("🤖 Train Model", use_container_width=True, help="Train the Bayesian Network model"):
            if train_model():
                accuracy = st.session_state.training_results.get('accuracy', 0) * 100
                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.1f}%")
                st.rerun()
            else:
                st.error("❌ Failed to train model")
    
    with col4:
        if st.button("🔄 Refresh Status", use_container_width=True, help="Refresh system status"):
            st.rerun()
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Prediction Section
        st.header("🎯 Lineup Efficiency Prediction")
        
        st.markdown("""
        <div class="talent-input">
        """, unsafe_allow_html=True)
        
        st.subheader("Rate Your Lineup's Talent Levels")
        
        scoring = st.selectbox(
            "🏀 Scoring Talent",
            ["Low", "Medium", "High", "Very High"],
            index=2,
            help="Points per game adjusted by true shooting percentage"
        )
        
        playmaking = st.selectbox(
            "🔗 Playmaking Talent", 
            ["Low", "Medium", "High", "Very High"],
            index=1,
            help="Assists per game"
        )
        
        rebounding = st.selectbox(
            "📊 Rebounding Talent",
            ["Low", "Medium", "High", "Very High"],
            index=1,
            help="Rebounds per game"
        )
        
        defensive = st.selectbox(
            "🛡️ Defensive Talent",
            ["Low", "Medium", "High", "Very High"],
            index=1,
            help="Combined steals and blocks per game"
        )
        
        if st.button("🎯 Predict Lineup Efficiency", use_container_width=True, type="primary"):
            if not st.session_state.model_trained:
                st.error("❌ Please train the model first using the 'Train Model' button above!")
            else:
                with st.spinner("🤔 Analyzing lineup..."):
                    features = {
                        'scoring': scoring,
                        'playmaking': playmaking,
                        'rebounding': rebounding,
                        'defensive': defensive
                    }
                    
                    result = st.session_state.model.predict(features)
                    st.session_state.last_prediction = result
                    
                    if result['success']:
                        prediction = result['prediction']
                        
                        # Display prediction result
                        st.markdown("""
                        <div class="prediction-result">
                        """, unsafe_allow_html=True)
                        
                        # Main prediction with color coding
                        efficiency_color = {
                            'Very Poor': 'red',
                            'Poor': 'orange', 
                            'Good': 'blue',
                            'Excellent': 'green'
                        }[prediction['predicted_efficiency']]
                        
                        st.markdown(f"""
                        <h3 style='color: {efficiency_color}; text-align: center;'>
                            🎯 Predicted Efficiency: <strong>{prediction['predicted_efficiency']}</strong>
                        </h3>
                        <p style='text-align: center; font-size: 1.1rem;'>
                            Confidence: <strong>{(prediction['probability'] * 100):.1f}%</strong>
                        </p>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown chart
                        fig_prob = create_probability_breakdown_plot(prediction)
                        if fig_prob:
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display talent summary
                        st.subheader("📋 Your Lineup Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        talent_icons = {
                            'scoring': '🏀',
                            'playmaking': '🔗',
                            'rebounding': '📊',
                            'defensive': '🛡️'
                        }
                        
                        with col1:
                            st.metric("Scoring", scoring, delta=None, delta_color="off")
                        with col2:
                            st.metric("Playmaking", playmaking, delta=None, delta_color="off")
                        with col3:
                            st.metric("Rebounding", rebounding, delta=None, delta_color="off")
                        with col4:
                            st.metric("Defense", defensive, delta=None, delta_color="off")
                            
                    else:
                        st.error(f"❌ Prediction failed: {result['message']}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Visualization and Information Section
        if st.session_state.data_loaded:
            display_data_insights()
        else:
            st.info("💡 Load data to see insights and visualizations")
        
        # About Section
        st.markdown("---")
        st.header("📚 About This System")
        
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 How It Works</h4>
        <p>This system uses <strong>Bayesian Networks</strong> to predict NBA lineup efficiency based on four key talent dimensions:</p>
        
        <h5>🏀 Scoring Talent</h5>
        <ul>
            <li>Points per game adjusted by true shooting percentage</li>
            <li>Shooting efficiency and scoring volume</li>
        </ul>
        
        <h5>🔗 Playmaking Talent</h5>
        <ul>
            <li>Assists per game</li>
            <li>Court vision and passing ability</li>
        </ul>
        
        <h5>📊 Rebounding Talent</h5>
        <ul>
            <li>Rebounds per game</li>
            <li>Offensive and defensive rebounding</li>
        </ul>
        
        <h5>🛡️ Defensive Talent</h5>
        <ul>
            <li>Combined steals and blocks per game</li>
            <li>Defensive impact and versatility</li>
        </ul>
        
        <p><strong>📈 Model Training:</strong> The Bayesian Network learns probabilistic relationships between 
        talent levels and lineup efficiency from historical NBA data.</p>
        
        <p><strong>🎯 Prediction:</strong> For any given combination of talent levels, the model calculates 
        the most probable efficiency outcome along with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "NBA Lineup Efficiency Predictor • Built with Streamlit • "
        "Data: Simulated NBA Lineup Statistics"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()