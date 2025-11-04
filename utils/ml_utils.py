"""
Enhanced Machine Learning Utilities for NBA Lineup Efficiency Prediction
Bayesian Network with improved probability estimation and ensemble support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBayesianNetwork:
    """Enhanced Bayesian Network with better probability estimation and smoothing."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize Enhanced Bayesian Network.
        
        Args:
            alpha: Laplace smoothing parameter to handle unseen combinations
        """
        self.model_trained = False
        self.probability_tables = None
        self.preprocessing_metadata = None
        self.accuracy = 0.0
        self.alpha = alpha  # Laplace smoothing parameter
        self.feature_importance = None
        
    def train_model(self, df: pd.DataFrame, preprocessing_metadata: Dict) -> Dict:
        """
        Train Enhanced Bayesian Network with better probability estimation.
        
        Args:
            df: Preprocessed DataFrame
            preprocessing_metadata: Metadata for preprocessing
            
        Returns:
            Training results dictionary
        """
        logger.info("Training Enhanced Bayesian Network model...")
        
        try:
            feature_cols = preprocessing_metadata['feature_columns']
            target_col = preprocessing_metadata['target_column']
            
            # Build enhanced probability tables with smoothing
            self.probability_tables = self._build_enhanced_probability_tables(
                df, feature_cols, target_col
            )
            
            self.preprocessing_metadata = preprocessing_metadata
            
            # Evaluate model with proper train-test split
            X = df[feature_cols]
            y = df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train on training set
            train_df = pd.concat([X_train, y_train], axis=1)
            self.probability_tables = self._build_enhanced_probability_tables(
                train_df, feature_cols, target_col
            )
            
            # Evaluate on test set
            predictions = []
            for _, row in X_test.iterrows():
                pred = self._predict_single_enhanced(row)
                predictions.append(pred)
            
            self.accuracy = accuracy_score(y_test, predictions)
            self.model_trained = True
            
            # Calculate feature importance
            self.feature_importance = self._calculate_feature_importance(X_test, y_test)
            
            # Additional metrics
            conf_matrix = confusion_matrix(y_test, predictions)
            class_report = classification_report(y_test, predictions, output_dict=True)
            
            results = {
                'accuracy': self.accuracy,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'feature_importance': self.feature_importance,
                'probability_tables_summary': {
                    k: f"{len(v)} entries" for k, v in self.probability_tables.items()
                }
            }
            
            logger.info(f"Enhanced Bayesian Network training completed with accuracy: {self.accuracy:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training enhanced Bayesian model: {e}")
            raise
    
    def _build_enhanced_probability_tables(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
        """Build enhanced probability tables with Laplace smoothing."""
        probability_tables = {}
        
        # Calculate prior probabilities with smoothing
        target_counts = df[target_col].value_counts()
        total_target = len(df)
        num_classes = len(target_counts)
        
        target_probs = {}
        for target_val in range(num_classes):
            count = target_counts.get(target_val, 0)
            # Laplace smoothing: (count + alpha) / (total + alpha * num_classes)
            target_probs[target_val] = (count + self.alpha) / (total_target + self.alpha * num_classes)
        
        probability_tables['prior'] = target_probs
        probability_tables['target_counts'] = target_counts.to_dict()
        probability_tables['total_instances'] = total_target
        
        # Calculate conditional probabilities for each feature given target
        for feature in feature_cols:
            conditional_probs = {}
            feature_values = sorted(df[feature].unique())
            num_feature_values = len(feature_values)
            
            for target_val in range(num_classes):
                subset = df[df[target_col] == target_val]
                target_count = len(subset)
                
                feature_probs = {}
                for feature_val in feature_values:
                    count = len(subset[subset[feature] == feature_val])
                    # Laplace smoothing for conditional probabilities
                    prob = (count + self.alpha) / (target_count + self.alpha * num_feature_values)
                    feature_probs[feature_val] = prob
                
                conditional_probs[target_val] = feature_probs
            
            probability_tables[feature] = conditional_probs
        
        return probability_tables
    
    def _predict_single_enhanced(self, features: pd.Series) -> int:
        """Enhanced prediction with better probability handling."""
        if not self.model_trained:
            raise ValueError("Model not trained")
        
        # Calculate posterior probabilities for each target class
        posterior_probs = {}
        total_instances = self.probability_tables['total_instances']
        
        for target_val, prior_prob in self.probability_tables['prior'].items():
            likelihood = 1.0
            
            for feature, feature_val in features.items():
                if (feature in self.probability_tables and 
                    target_val in self.probability_tables[feature] and
                    feature_val in self.probability_tables[feature][target_val]):
                    
                    likelihood *= self.probability_tables[feature][target_val][feature_val]
                else:
                    # Use a very small probability for completely unseen combinations
                    likelihood *= 1e-6
            
            posterior_probs[target_val] = prior_prob * likelihood
        
        # Normalize probabilities
        total = sum(posterior_probs.values())
        if total > 0:
            for key in posterior_probs:
                posterior_probs[key] /= total
        else:
            # If all probabilities are zero, use uniform distribution
            for key in posterior_probs:
                posterior_probs[key] = 1.0 / len(posterior_probs)
        
        # Return prediction with highest probability
        return max(posterior_probs.items(), key=lambda x: x[1])[0]
    
    def _calculate_feature_importance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Calculate feature importance using permutation importance."""
        baseline_accuracy = self.accuracy
        feature_importance = {}
        
        for feature in X_test.columns:
            # Shuffle the feature
            X_perturbed = X_test.copy()
            X_perturbed[feature] = np.random.permutation(X_perturbed[feature].values)
            
            # Calculate accuracy with perturbed feature
            predictions = []
            for _, row in X_perturbed.iterrows():
                pred = self._predict_single_enhanced(row)
                predictions.append(pred)
            
            perturbed_accuracy = accuracy_score(y_test, predictions)
            importance_score = baseline_accuracy - perturbed_accuracy
            feature_importance[feature] = max(importance_score, 0)  # Ensure non-negative
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        return feature_importance
    
    def predict(self, features_dict: Dict) -> Dict[str, Any]:
        """
        Predict lineup efficiency with enhanced probability estimation.
        
        Args:
            features_dict: Dictionary with talent levels
                {
                    'scoring': 'Low'|'Medium'|'High'|'Very High',
                    'playmaking': 'Low'|'Medium'|'High'|'Very High', 
                    'rebounding': 'Low'|'Medium'|'High'|'Very High',
                    'defensive': 'Low'|'Medium'|'High'|'Very High'
                }
        
        Returns:
            Dictionary with prediction results
        """
        if not self.model_trained:
            return {'success': False, 'message': 'Model not trained'}
        
        try:
            # Convert input to encoded features
            talent_mapping = self.preprocessing_metadata['talent_mapping']
            efficiency_mapping = self.preprocessing_metadata['efficiency_mapping']
            reverse_efficiency_mapping = {v: k for k, v in efficiency_mapping.items()}
            
            encoded_features = {
                'scoring_talent_encoded': talent_mapping[features_dict['scoring']],
                'playmaking_talent_encoded': talent_mapping[features_dict['playmaking']],
                'rebounding_talent_encoded': talent_mapping[features_dict['rebounding']],
                'defensive_talent_encoded': talent_mapping[features_dict['defensive']]
            }
            
            # Calculate probabilities for all target classes
            posterior_probs = {}
            features_series = pd.Series(encoded_features)
            
            total_instances = self.probability_tables['total_instances']
            
            for target_val, prior_prob in self.probability_tables['prior'].items():
                likelihood = 1.0
                
                for feature, feature_val in features_series.items():
                    if (feature in self.probability_tables and 
                        target_val in self.probability_tables[feature] and
                        feature_val in self.probability_tables[feature][target_val]):
                        
                        likelihood *= self.probability_tables[feature][target_val][feature_val]
                    else:
                        # Use smoothed probability for unseen combinations
                        num_feature_values = len(self.probability_tables[feature][target_val])
                        likelihood *= self.alpha / (self.probability_tables['target_counts'].get(target_val, 1) + self.alpha * num_feature_values)
                
                posterior_probs[target_val] = prior_prob * likelihood
            
            # Normalize probabilities
            total = sum(posterior_probs.values())
            if total > 0:
                for key in posterior_probs:
                    posterior_probs[key] /= total
            else:
                # If all probabilities are zero, use uniform distribution
                for key in posterior_probs:
                    posterior_probs[key] = 1.0 / len(posterior_probs)
            
            # Get prediction
            predicted_class = max(posterior_probs.items(), key=lambda x: x[1])[0]
            predicted_efficiency = reverse_efficiency_mapping[predicted_class]
            
            # Convert all probabilities to readable format
            all_probabilities = {
                reverse_efficiency_mapping[k]: float(v) for k, v in posterior_probs.items()
            }
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(posterior_probs)
            
            return {
                'success': True,
                'prediction': {
                    'predicted_efficiency': predicted_efficiency,
                    'probability': float(posterior_probs[predicted_class]),
                    'all_probabilities': all_probabilities,
                    'model_type': 'Enhanced Bayesian Network',
                    'confidence_metrics': confidence_metrics,
                    'feature_contributions': self._get_feature_contributions(encoded_features, posterior_probs)
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced Bayesian prediction error: {e}")
            return {'success': False, 'message': f'Prediction error: {str(e)}'}
    
    def _calculate_confidence_metrics(self, posterior_probs: Dict) -> Dict:
        """Calculate confidence metrics for the prediction."""
        probs = list(posterior_probs.values())
        max_prob = max(probs)
        second_max_prob = sorted(probs)[-2] if len(probs) > 1 else 0
        
        return {
            'max_probability': float(max_prob),
            'margin': float(max_prob - second_max_prob),
            'entropy': float(-sum(p * np.log(p + 1e-10) for p in probs)),
            'confidence_ratio': float(max_prob / (second_max_prob + 1e-10)) if second_max_prob > 0 else float('inf')
        }
    
    def _get_feature_contributions(self, encoded_features: Dict, posterior_probs: Dict) -> Dict:
        """Calculate contribution of each feature to the prediction."""
        if self.feature_importance is None:
            return {}
        
        contributions = {}
        total_importance = sum(self.feature_importance.values())
        
        if total_importance > 0:
            for feature, importance in self.feature_importance.items():
                feature_name = feature.replace('_encoded', '').replace('_talent', '').title()
                contributions[feature_name] = {
                    'importance': float(importance),
                    'value': encoded_features[feature],
                    'contribution_score': float(importance * 100)  # Percentage contribution
                }
        
        return contributions
    
    def predict_with_metadata(self, features_dict: Dict) -> Dict[str, Any]:
        """
        Predict lineup efficiency with additional metadata for ensemble.
        Same as predict() but returns consistent format for ensemble.
        """
        return self.predict(features_dict)
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about the trained model."""
        if not self.model_trained:
            return {'success': False, 'message': 'Model not trained'}
        
        try:
            insights = {
                'model_type': 'Enhanced Bayesian Network',
                'accuracy': self.accuracy,
                'training_instances': self.probability_tables.get('total_instances', 0),
                'feature_importance': self.feature_importance,
                'laplace_smoothing_alpha': self.alpha,
                'target_distribution': self.probability_tables.get('target_counts', {})
            }
            
            # Calculate most influential features
            if self.feature_importance:
                top_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                insights['top_features'] = [
                    {'feature': feat.replace('_encoded', '').replace('_talent', '').title(), 'importance': imp}
                    for feat, imp in top_features
                ]
            
            return {'success': True, 'insights': insights}
            
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return {'success': False, 'message': str(e)}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.model_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'probability_tables': self.probability_tables,
            'preprocessing_metadata': self.preprocessing_metadata,
            'accuracy': self.accuracy,
            'feature_importance': self.feature_importance,
            'alpha': self.alpha,
            'model_type': 'EnhancedBayesianNetwork'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Enhanced Bayesian model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        if model_data.get('model_type') != 'EnhancedBayesianNetwork':
            logger.warning("Loaded model may not be compatible with EnhancedBayesianNetwork")
        
        self.probability_tables = model_data['probability_tables']
        self.preprocessing_metadata = model_data['preprocessing_metadata']
        self.accuracy = model_data['accuracy']
        self.feature_importance = model_data.get('feature_importance')
        self.alpha = model_data.get('alpha', 0.1)
        self.model_trained = True
        
        logger.info(f"Enhanced Bayesian model loaded from {filepath}")


class ModelEnsemble:
    """Ensemble class to combine Bayesian and CNN-LSTM predictions."""
    
    def __init__(self):
        self.bayesian_model = None
        self.cnn_lstm_model = None
        self.ensemble_weights = {'bayesian': 0.4, 'cnn_lstm': 0.6}
    
    def set_models(self, bayesian_model: EnhancedBayesianNetwork, cnn_lstm_model: Any):
        """Set the models for ensemble prediction."""
        self.bayesian_model = bayesian_model
        self.cnn_lstm_model = cnn_lstm_model
    
    def set_weights(self, bayesian_weight: float, cnn_lstm_weight: float):
        """Set ensemble weights."""
        total = bayesian_weight + cnn_lstm_weight
        self.ensemble_weights = {
            'bayesian': bayesian_weight / total,
            'cnn_lstm': cnn_lstm_weight / total
        }
    
    def predict_ensemble(self, features_dict: Dict, sequence_data: np.ndarray = None) -> Dict[str, Any]:
        """Make ensemble prediction combining both models."""
        if not self.bayesian_model or not self.cnn_lstm_model:
            return {'success': False, 'message': 'Both models not available for ensemble'}
        
        if not self.bayesian_model.model_trained or not self.cnn_lstm_model.model_trained:
            return {'success': False, 'message': 'Both models not trained'}
        
        try:
            # Get Bayesian prediction
            bayesian_result = self.bayesian_model.predict(features_dict)
            
            # Get CNN-LSTM prediction
            cnn_lstm_result = self.cnn_lstm_model.predict_single(sequence_data) if sequence_data is not None else {
                'success': False, 'message': 'No sequence data provided'
            }
            
            if not bayesian_result['success'] or not cnn_lstm_result['success']:
                # Return whichever prediction is available
                if bayesian_result['success']:
                    return bayesian_result
                elif cnn_lstm_result['success']:
                    return cnn_lstm_result
                else:
                    return {'success': False, 'message': 'Both models failed to predict'}
            
            # Combine predictions using ensemble weights
            bayesian_probs = bayesian_result['prediction']['all_probabilities']
            cnn_lstm_probs = cnn_lstm_result['prediction']['all_probabilities']
            
            ensemble_probs = {}
            for efficiency in ['Very Poor', 'Poor', 'Good', 'Excellent']:
                ensemble_probs[efficiency] = (
                    self.ensemble_weights['bayesian'] * bayesian_probs[efficiency] +
                    self.ensemble_weights['cnn_lstm'] * cnn_lstm_probs[efficiency]
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
                    'model_type': 'Ensemble (Enhanced Bayesian + CNN-LSTM)',
                    'component_predictions': {
                        'bayesian': bayesian_result['prediction'],
                        'cnn_lstm': cnn_lstm_result['prediction']
                    },
                    'ensemble_weights': self.ensemble_weights
                }
            }
        
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return {'success': False, 'message': f'Ensemble prediction error: {str(e)}'}


# For backward compatibility
BayesianNetworkModel = EnhancedBayesianNetwork