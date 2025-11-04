"""
CNN-LSTM Model for NBA Lineup Sequence Prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNLSTMModel:
    """CNN-LSTM model for sequential NBA lineup prediction."""
    
    def __init__(self):
        self.model_trained = False
        self.model = None
        self.sequence_length = 10
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = None
        self.accuracy = 0.0
        
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN-LSTM training."""
        sequences = []
        targets = []
        
        # Convert categorical features to numerical
        feature_cols = ['scoring_talent_encoded', 'playmaking_talent_encoded', 
                       'rebounding_talent_encoded', 'defensive_talent_encoded']
        
        # Create sequences
        for i in range(len(data) - sequence_length):
            seq = data[feature_cols].iloc[i:i + sequence_length].values
            target = data['efficiency_encoded'].iloc[i + sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape: Tuple[int, int], num_classes: int) -> Sequential:
        """Build CNN-LSTM model architecture."""
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # LSTM layers for sequence learning
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(50, dropout=0.2, recurrent_dropout=0.2),
            
            # Dense layers for classification
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, df: pd.DataFrame, preprocessing_metadata: Dict) -> Dict:
        """
        Train CNN-LSTM model on sequential NBA data.
        Returns training results.
        """
        logger.info("Training CNN-LSTM model...")
        
        try:
            # Create sequences for training
            sequences, targets = self.create_sequences(df, self.sequence_length)
            
            if len(sequences) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Encode targets
            targets_encoded = to_categorical(targets)
            num_classes = targets_encoded.shape[1]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets_encoded, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Build and train model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape, num_classes)
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1,
                shuffle=True
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            self.accuracy = test_accuracy
            self.model_trained = True
            
            # Get predictions for additional metrics
            y_pred = self.model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            results = {
                'accuracy': test_accuracy,
                'loss': test_loss,
                'training_history': self.history.history,
                'model_summary': self._get_model_summary(),
                'sequence_length': self.sequence_length,
                'num_sequences': len(sequences)
            }
            
            logger.info(f"CNN-LSTM model training completed with accuracy: {test_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training CNN-LSTM model: {e}")
            raise
    
    def _get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)
    
    def predict_single(self, sequence_data: np.ndarray) -> Dict[str, Any]:
        """Predict efficiency for a single sequence."""
        if not self.model_trained:
            return {'success': False, 'message': 'CNN-LSTM model not trained'}
        
        try:
            # Ensure sequence has correct shape
            if sequence_data.shape != (1, self.sequence_length, 4):
                raise ValueError(f"Expected shape (1, {self.sequence_length}, 4), got {sequence_data.shape}")
            
            # Make prediction
            prediction = self.model.predict(sequence_data, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
            probability = np.max(prediction, axis=1)[0]
            
            # Convert back to efficiency label
            efficiency_mapping = {0: 'Very Poor', 1: 'Poor', 2: 'Good', 3: 'Excellent'}
            predicted_efficiency = efficiency_mapping.get(predicted_class, 'Unknown')
            
            # Get all probabilities
            all_probabilities = {
                efficiency_mapping[i]: float(prediction[0][i]) 
                for i in range(len(prediction[0]))
            }
            
            return {
                'success': True,
                'prediction': {
                    'predicted_efficiency': predicted_efficiency,
                    'probability': float(probability),
                    'all_probabilities': all_probabilities,
                    'model_type': 'CNN-LSTM'
                }
            }
            
        except Exception as e:
            logger.error(f"CNN-LSTM prediction error: {e}")
            return {'success': False, 'message': f'Prediction error: {str(e)}'}
    
    def create_sequence_from_talents(self, talents: List[Dict]) -> np.ndarray:
        """Create sequence from list of talent dictionaries."""
        talent_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
        
        sequence_data = []
        for talent in talents:
            seq_point = [
                talent_mapping[talent['scoring']],
                talent_mapping[talent['playmaking']],
                talent_mapping[talent['rebounding']],
                talent_mapping[talent['defensive']]
            ]
            sequence_data.append(seq_point)
        
        # Pad or truncate to sequence length
        if len(sequence_data) > self.sequence_length:
            sequence_data = sequence_data[-self.sequence_length:]
        elif len(sequence_data) < self.sequence_length:
            # Pad with zeros
            padding = [[0, 0, 0, 0]] * (self.sequence_length - len(sequence_data))
            sequence_data = padding + sequence_data
        
        return np.array([sequence_data])
    
    def ensemble_predict(self, bayesian_pred: Dict, cnn_lstm_pred: Dict) -> Dict[str, Any]:
        """Combine Bayesian and CNN-LSTM predictions using ensemble learning."""
        if not bayesian_pred['success'] or not cnn_lstm_pred['success']:
            return bayesian_pred if bayesian_pred['success'] else cnn_lstm_pred
        
        bayesian_probs = bayesian_pred['prediction']['all_probabilities']
        cnn_lstm_probs = cnn_lstm_pred['prediction']['all_probabilities']
        
        # Weighted average (you can adjust weights based on model performance)
        weights = {'bayesian': 0.4, 'cnn_lstm': 0.6}
        
        ensemble_probs = {}
        for efficiency in ['Very Poor', 'Poor', 'Good', 'Excellent']:
            ensemble_probs[efficiency] = (
                weights['bayesian'] * bayesian_probs[efficiency] +
                weights['cnn_lstm'] * cnn_lstm_probs[efficiency]
            )
        
        # Get final prediction
        predicted_efficiency = max(ensemble_probs.items(), key=lambda x: x[1])[0]
        probability = ensemble_probs[predicted_efficiency]
        
        return {
            'success': True,
            'prediction': {
                'predicted_efficiency': predicted_efficiency,
                'probability': probability,
                'all_probabilities': ensemble_probs,
                'model_type': 'Ensemble (Bayesian + CNN-LSTM)',
                'component_predictions': {
                    'bayesian': bayesian_pred['prediction'],
                    'cnn_lstm': cnn_lstm_pred['prediction']
                }
            }
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.model_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model_weights': self.model.get_weights(),
            'sequence_length': self.sequence_length,
            'accuracy': self.accuracy,
            'history': self.history.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"CNN-LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str, model_architecture: Sequential) -> None:
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model_architecture.set_weights(model_data['model_weights'])
        self.model = model_architecture
        self.sequence_length = model_data['sequence_length']
        self.accuracy = model_data['accuracy']
        self.model_trained = True
        
        logger.info(f"CNN-LSTM model loaded from {filepath}")