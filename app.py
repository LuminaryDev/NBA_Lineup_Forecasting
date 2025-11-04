#!/usr/bin/env python3
"""
NBA Lineup Efficiency Predictor - Streamlit App with Bayesian + LSTM Ensemble
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
    from lstm_utils import LSTMModel  # ADAPTED: LSTM instead of CNN-LSTM
except ImportError as e:
    st.error(f"Import error: {e}. Make sure utils/ directory exists with all required files")
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
        background: black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
    }
    .model-comparison {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'bayesian_model' not in st.session_state:
        st.session_state.bayesian_model = BayesianNetworkModel()
    if 'lstm_model' not in st.session_state:  # ADAPTED: LSTM
        st.session_state.lstm_model = LSTMModel()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'bayesian_trained' not in st.session_state:
        st.session_state.bayesian_trained = False
    if 'lstm_trained' not in st.session_state:  # ADAPTED
        st.session_state.lstm_trained = False
    if 'data_stats' not in st.session_state:
        st.session_state.data_stats = {}
    if 'bayesian_results' not in st.session_state:
        st.session_state.bayesian_results = {}
    if 'lstm_results' not in st.session_state:  # ADAPTED
        st.session_state.lstm_results = {}
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'show_success' not in st.session_state:
        st.session_state.show_success = False
    if 'success_message' not in st.session_state:
        st.session_state.success_message = ""
    if 'show_error' not in st.session_state:
        st.session_state.show_error = False
    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""
    if 'prediction_history' not in st.session_state:  # ADAPTED: For LSTM sequences
        st.session_state.prediction_history = []

def update_status():
    """Update system status"""
    return {
        'data_loaded': st.session_state.data_loaded,
        'bayesian_trained': st.session_state.bayesian_trained,
        'lstm_trained': st.session_state.lstm_trained,  # ADAPTED
        'data_count': len(st.session_state.data_processor.processed_data) if st.session_state.data_loaded and hasattr(st.session_state.data_processor, 'processed_data') and st.session_state.data_processor.processed_data is not None else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'bayesian_accuracy': st.session_state.bayesian_results.get('accuracy', 0) if st.session_state.bayesian_trained else 0,
        'lstm_accuracy': st.session_state.lstm_results.get('accuracy', 0) if st.session_state.lstm_trained else 0  # ADAPTED
    }

def load_data():
    """Load existing data"""
    try:
        with st.spinner("Loading sample NBA data..."):
            df = st.session_state.data_processor.load_sample_data()
            st.session_state.data_loaded = True
            st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
            st.session_state.show_success = True
            st.session_state.success_message = "✅ Data loaded successfully!"
            return True
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.session_state.show_error = True
        st.session_state.error_message = f"❌ Failed to load data: {str(e)}"
        return False

def download_new_data():
    """Download new data from NBA API"""
    try:
        with st.spinner("Downloading new NBA data..."):
            df = st.session_state.data_processor.download_nba_data()
            st.session_state.data_loaded = True
            st.session_state.data_stats = st.session_state.data_processor.get_data_stats()
            st.session_state.show_success = True
            st.session_state.success_message = "✅ New data downloaded successfully!"
            return True
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        st.session_state.show_error = True
        st.session_state.error_message = f"❌ Failed to download data: {str(e)}"
        return False

def train_bayesian_model():
    """Train the Bayesian Network model"""
    if not st.session_state.data_loaded:
        st.session_state.show_error = True
        st.session_state.error_message = "❌ Please load data first before training the model!"
        return False
    
    try:
        with st.spinner("Training Bayesian Network model..."):
            if hasattr(st.session_state.data_processor, 'processed_data') and st.session_state.data_processor.processed_data is not None:
                processed_df, preprocessing_metadata = st.session_state.data_processor.preprocess_data(
                    st.session_state.data_processor.processed_data
                )
                
                results = st.session_state.bayesian_model.train_model(processed_df, preprocessing_metadata)
                st.session_state.bayesian_trained = True
                st.session_state.bayesian_results = results
                st.session_state.show_success = True
                accuracy = results.get('accuracy', 0) * 100
                st.session_state.success_message = f"✅ Bayesian Model trained successfully! Accuracy: {accuracy:.1f}%"
                return True
            else:
                st.session_state.show_error = True
                st.session_state.error_message = "❌ No data available for training. Please load data first."
                return False
    except Exception as e:
        logger.error(f"Error training Bayesian model: {e}")
        st.session_state.show_error = True
        st.session_state.error_message = f"❌ Failed to train Bayesian model: {str(e)}"
        return False

def train_lstm_model():  # ADAPTED: LSTM instead of CNN-LSTM
    """Train the LSTM model"""
    if not st.session_state.data_loaded:
        st.session_state.show_error = True
        st.session_state.error_message = "❌ Please load data first before training the model!"
        return False
    
    try:
        with st.spinner("Training LSTM model (this may take a while)..."):
            if hasattr(st.session_state.data_processor, 'processed_data') and st.session_state.data_processor.processed_data is not None:
                processed_df, preprocessing_metadata = st.session_state.data_processor.preprocess_data(
                    st.session_state.data_processor.processed_data
                )
                
                results = st.session_state.lstm_model.train_model(processed_df, preprocessing_metadata)
                st.session_state.lstm_trained = True
                st.session_state.lstm_results = results
                st.session_state.show_success = True
                accuracy = results.get('accuracy', 0) * 100
                st.session_state.success_message = f"✅ LSTM Model trained successfully! Accuracy: {accuracy:.1f}%"
                return True
            else:
                st.session_state.show_error = True
                st.session_state.error_message = "❌ No data available for training. Please load data first."
                return False
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        st.session_state.show_error = True
        st.session_state.error_message = f"❌ Failed to train LSTM model: {str(e)}"
        return False

def train_all_models():  # ADAPTED
    """Train both Bayesian and LSTM models"""
    if not st.session_state.data_loaded:
        st.session_state.show_error = True
        st.session_state.error_message = "❌ Please load data first before training models!"
        return False
    
    success_count = 0
    if train_bayesian_model():
        success_count += 1
    if train_lstm_model():
        success_count += 1
    
    if success_count == 2:
        st.session_state.show_success = True
        st.session_state.success_message = "✅ Both models trained successfully!"
        return True
    elif success_count == 1:
        st.session_state.show_success = True
        st.session_state.success_message = "⚠️ One model trained successfully, one failed"
        return False
    else:
        return False

def create_model_comparison_plot():  # ADAPTED
    """Create model accuracy comparison plot"""
    status = update_status()
    
    models = ['Bayesian Network', 'LSTM']
    accuracies = [
        status['bayesian_accuracy'] * 100,
        status['lstm_accuracy'] * 100
    ]
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = px.bar(
        x=models,
        y=accuracies,
        title='Model Accuracy Comparison',
        labels={'x': 'Model', 'y': 'Accuracy (%)'},
        color=models,
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        showlegend=False
    )
    
    return fig

def create_training_history_plot():  # ADAPTED
    """Create LSTM training history plot"""
    if not st.session_state.lstm_trained:
        return None
    
    try:
        history = st.session_state.lstm_results.get('training_history', {})
        if not history:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history['accuracy'],
            mode='lines',
            name='Training Accuracy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=history['val_accuracy'],
            mode='lines',
            name='Validation Accuracy',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='LSTM Training History',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1]
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating training history plot: {e}")
        return None


def create_probability_breakdown_plot(prediction: dict):
    """Create a probability breakdown bar chart for the prediction.

    Accepts a prediction dict and tries to extract a mapping of labels -> probabilities.
    Handles multiple possible keys and missing data gracefully.
    """
    try:
        # Try common keys
        probs = None
        if isinstance(prediction, dict):
            # Some predictors return prediction['all_probabilities']
            if 'all_probabilities' in prediction:
                probs = prediction['all_probabilities']
            # Or top-level probabilities mapping
            elif 'probabilities' in prediction:
                probs = prediction['probabilities']
            # Or prediction may be nested under 'prediction'
            elif 'prediction' in prediction and isinstance(prediction['prediction'], dict):
                inner = prediction['prediction']
                if 'all_probabilities' in inner:
                    probs = inner['all_probabilities']
                elif 'probabilities' in inner:
                    probs = inner['probabilities']

        if not probs:
            return None

        labels = list(probs.keys())
        values = [float(probs[k]) * 100.0 for k in labels]

        fig = go.Figure([go.Bar(x=labels, y=values, marker_color=['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c'])])
        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability (%)',
            xaxis_title='Outcome',
            yaxis_range=[0, 100]
        )
        return fig

    except Exception as e:
        logger.error(f"Error creating probability breakdown plot: {e}")
        return None


def display_data_insights():
    """Render dataset summary, sample rows and basic visualizations in the right column.

    This function is referenced from the UI. If data isn't available, it shows a helpful message.
    """
    try:
        st.header("📊 Data Insights")

        if not st.session_state.data_loaded:
            st.info("No data loaded. Use the 'Load Sample Data' or 'Download New Data' buttons.")
            return

        # Basic stats
        stats = st.session_state.data_stats or {}
        with st.expander("Dataset Summary", expanded=True):
            if stats:
                for k, v in stats.items():
                    st.write(f"**{k}**: {v}")
            else:
                st.write("No summary statistics available.")

        # Show sample rows if present
        if hasattr(st.session_state.data_processor, 'processed_data') and st.session_state.data_processor.processed_data is not None:
            df = st.session_state.data_processor.processed_data
            with st.expander("Sample Rows", expanded=False):
                try:
                    st.dataframe(df.head(10))
                except Exception:
                    st.write("Unable to display sample rows.")

            # Attempt simple visualizations for expected columns (ADAPTED to system features)
            numeric_cols = []
            for col in ['PLUS_MINUS', 'FG_PCT', 'AST', 'REB']:  # System-specific
                if col in df.columns:
                    numeric_cols.append(col)

            if numeric_cols:
                with st.expander("Feature Distribution Plots", expanded=True):
                    for col in numeric_cols:
                        try:
                            fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=10)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.write(f"Could not plot distribution for {col}")
        else:
            st.info("Data processor has no processed_data attribute or it's empty.")

    except Exception as e:
        logger.error(f"Error displaying data insights: {e}")
        st.error("An error occurred while rendering data insights.")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>🏀 NBA Lineup Efficiency Predictor</h1>
        <p class="lead" style="font-size: 1.2rem; margin-bottom: 0;">Using Bayesian Networks + LSTM Ensemble to predict NBA lineup performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display success/error messages
    if st.session_state.show_success:
        st.markdown(f'<div class="success-message">{st.session_state.success_message}</div>', unsafe_allow_html=True)
        if st.button("OK", key="clear_success"):
            st.session_state.show_success = False
            st.rerun()
    
    if st.session_state.show_error:
        st.markdown(f'<div class="error-message">{st.session_state.error_message}</div>', unsafe_allow_html=True)
        if st.button("OK", key="clear_error"):
            st.session_state.show_error = False
            st.rerun()
    
    # Status Section
    st.header("📈 System Status")
    
    status = update_status()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        status_color = "🟢" if status['bayesian_trained'] else "🔴"
        st.markdown(f"""
        <div class="status-card">
            <h3>{status_color}</h3>
            <strong>Bayesian Model</strong><br>
            <span style="color: {'green' if status['bayesian_trained'] else 'red'}">
                {'Trained' if status['bayesian_trained'] else 'Not Trained'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_color = "🟢" if status['lstm_trained'] else "🔴"  # ADAPTED
        st.markdown(f"""
        <div class="status-card">
            <h3>{status_color}</h3>
            <strong>LSTM Model</strong><br>
            <span style="color: {'green' if status['lstm_trained'] else 'red'}">
                {'Trained' if status['lstm_trained'] else 'Not Trained'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="status-card">
            <h3>📊</h3>
            <strong>Data Points</strong><br>
            {status['data_count']:,}
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="status-card">
            <h3>⏰</h3>
            <strong>Last Update</strong><br>
            {status['timestamp']}
        </div>
        """, unsafe_allow_html=True)
    
    # Control Section
    st.header("🎮 Data & Model Controls")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📥 Load Sample Data", use_container_width=True, help="Load sample NBA lineup data"):
            load_data()
            st.rerun()
    
    with col2:
        if st.button("🔄 Download New Data", use_container_width=True, help="Download fresh data from NBA API"):
            download_new_data()
            st.rerun()
    
    with col3:
        if st.button("🤖 Train Bayesian", use_container_width=True, help="Train the Bayesian Network model"):
            train_bayesian_model()
            st.rerun()
    
    with col4:
        if st.button("🧠 Train LSTM", use_container_width=True, help="Train the LSTM model (takes longer)"):  # ADAPTED
            train_lstm_model()
            st.rerun()
    
    with col5:
        if st.button("🚀 Train All Models", use_container_width=True, help="Train both models"):
            train_all_models()
            st.rerun()
    
    # Model Comparison Section
    if st.session_state.bayesian_trained or st.session_state.lstm_trained:  # ADAPTED
        st.header("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_comparison = create_model_comparison_plot()
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            fig_training = create_training_history_plot()
            if fig_training:
                st.plotly_chart(fig_training, use_container_width=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Prediction Section
        st.header("🎯 Lineup Efficiency Prediction")
        
        st.markdown("""
        <div class="talent-input">
        """, unsafe_allow_html=True)
        
        st.subheader("Set Feature Levels (0=Low, 3=High)")
        
        proj = st.selectbox(
            "🔮 Projection Strength Level",
            [0, 1, 2, 3],
            index=2,
            help="LSTM-based projection strength"
        )
        
        fg = st.selectbox(
            "🎯 FG% Level", 
            [0, 1, 2, 3],
            index=2,
            help="Field goal percentage level"
        )
        
        pm = st.selectbox(
            "📈 Plus/Minus Level",
            [0, 1, 2, 3],
            index=2,
            help="On-court impact level"
        )
        
        net = st.selectbox(
            "⭐ Net Rating Talent Level",
            [0, 1, 2, 3],
            index=2,
            help="Lineup net rating talent"
        )
        
        ast = st.selectbox(
            "🤝 Avg Form Ratio AST Level",
            [0, 1, 2, 3],
            index=2,
            help="Average form ratio for assists"
        )
        
        # Model selection for prediction
        st.subheader("🎛️ Prediction Method")
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Ensemble (Recommended)", "Bayesian Network", "LSTM"],
            index=0,
            horizontal=True
        )
        
        if st.button("🎯 Predict Lineup Efficiency", use_container_width=True, type="primary"):
            if not st.session_state.bayesian_trained and not st.session_state.lstm_trained:  # ADAPTED
                st.session_state.show_error = True
                st.session_state.error_message = "❌ Please train at least one model first using the training buttons above!"
                st.rerun()
            else:
                with st.spinner("🤔 Analyzing lineup..."):
                    features = {
                        'PROJECTION_STRENGTH_LEVEL': proj,
                        'FG_PCT_LEVEL': fg,
                        'PLUS_MINUS_LEVEL': pm,
                        'LINEUP_NET_RATING_TALENT_LEVEL': net,
                        'AVG_FORM_RATIO_AST_LEVEL': ast
                    }
                    
                    # Store current prediction in history for sequence (ADAPTED for LSTM)
                    st.session_state.prediction_history.append(features)
                    if len(st.session_state.prediction_history) > 10:
                        st.session_state.prediction_history = st.session_state.prediction_history[-10:]
                    
                    # Make predictions based on selected method
                    if prediction_method == "Bayesian Network" and st.session_state.bayesian_trained:
                        result = st.session_state.bayesian_model.predict_with_metadata(features)
                    elif prediction_method == "LSTM" and st.session_state.lstm_trained:  # ADAPTED
                        # Create sequence from prediction history
                        sequence = st.session_state.lstm_model.create_sequence_from_talents(st.session_state.prediction_history)
                        result = st.session_state.lstm_model.predict_single(sequence)
                    else:  # Ensemble or fallback
                        bayesian_result = st.session_state.bayesian_model.predict_with_metadata(features) if st.session_state.bayesian_trained else None
                        lstm_result = None  # ADAPTED
                        
                        if st.session_state.lstm_trained:
                            sequence = st.session_state.lstm_model.create_sequence_from_talents(st.session_state.prediction_history)
                            lstm_result = st.session_state.lstm_model.predict_single(sequence)
                        
                        if bayesian_result and lstm_result and bayesian_result['success'] and lstm_result['success']:
                            result = st.session_state.lstm_model.ensemble_predict(bayesian_result, lstm_result)  # ADAPTED
                        elif bayesian_result and bayesian_result['success']:
                            result = bayesian_result
                        elif lstm_result and lstm_result['success']:
                            result = lstm_result
                        else:
                            result = {'success': False, 'message': 'No trained models available for prediction'}
                    
                    st.session_state.last_prediction = result
                    
                    if result['success']:
                        prediction = result['prediction']
                        
                        # Display prediction result
                        st.markdown("""
                        <div class="prediction-result">
                        """, unsafe_allow_html=True)
                        
                        # Main prediction with color coding
                        efficiency_color = {
                            'Low': 'red',
                            'Medium': 'orange', 
                            'High': 'green'
                        }[prediction['predicted_efficiency']]
                        
                        st.markdown(f"""
                        <h3 style='color: {efficiency_color}; text-align: center;'>
                            🎯 Predicted Efficiency: <strong>{prediction['predicted_efficiency']}</strong>
                        </h3>
                        <p style='text-align: center; font-size: 1.1rem;'>
                            Confidence: <strong>{(prediction['probability'] * 100):.1f}%</strong><br>
                            <small>Model: {prediction.get('model_type', 'Unknown')}</small>
                        </p>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown chart
                        fig_prob = create_probability_breakdown_plot(prediction)
                        if fig_prob:
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Show model comparison if ensemble
                        if prediction.get('model_type') == 'Ensemble (BN + LSTM)':  # ADAPTED
                            st.markdown("""
                            <div class="model-comparison">
                                <h5>🧠 Ensemble Component Predictions:</h5>
                            """, unsafe_allow_html=True)
                            
                            comp = prediction['component_predictions']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Bayesian Network", 
                                    comp['bayesian']['predicted_efficiency'],
                                    f"{comp['bayesian']['probability']*100:.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "LSTM",  # ADAPTED
                                    comp['lstm']['predicted_efficiency'],
                                    f"{comp['lstm']['probability']*100:.1f}%"
                                )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display feature summary (ADAPTED)
                        st.subheader("📋 Your Feature Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Projection Strength", proj, delta=None, delta_color="off")
                        with col2:
                            st.metric("FG%", fg, delta=None, delta_color="off")
                        with col3:
                            st.metric("Plus/Minus", pm, delta=None, delta_color="off")
                        with col4:
                            st.metric("Net Rating", net, delta=None, delta_color="off")
                        with col5:
                            st.metric("AST Form Ratio", ast, delta=None, delta_color="off")
                            
                    else:
                        st.session_state.show_error = True
                        st.session_state.error_message = f"❌ Prediction failed: {result['message']}"
                        st.rerun()
        
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
        <p>This system uses <strong>Ensemble Learning</strong> combining Bayesian Networks and LSTM to predict NBA lineup efficiency:</p>
        
        <h5>🤖 Bayesian Network</h5>
        <ul>
            <li>Probabilistic graphical model</li>
            <li>Fast inference and interpretable results</li>
            <li>Excellent for categorical data relationships</li>
        </ul>
        
        <h5>🧠 LSTM</h5>  <!-- ADAPTED -->
        <ul>
            <li>Deep learning recurrent network for time-series</li>
            <li>Captures temporal patterns in player sequences</li>
            <li>Better for forecasting player form trends</li>
        </ul>
        
        <h5>🚀 Ensemble Method</h5>
        <ul>
            <li>Combines strengths of both models</li>
            <li>Weighted average of predictions</li>
            <li>More robust and accurate results</li>
        </ul>
        
        <p><strong>Feature Dimensions:</strong></p>  <!-- ADAPTED -->
        <ul>
            <li><strong>🔮 Projection Strength:</strong> LSTM-based player forecasts</li>
            <li><strong>🎯 FG%:</strong> Shooting efficiency level</li>
            <li><strong>📈 Plus/Minus:</strong> On-court impact</li>
            <li><strong>⭐ Net Rating:</strong> Lineup talent rating</li>
            <li><strong>🤝 AST Form Ratio:</strong> Playmaking form from sequences</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "NBA Lineup Efficiency Predictor • Bayesian + LSTM Ensemble • "
        "Built with Streamlit • Data: NBA API 2023-24 Lineups"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
