# Complete Data Processing System for Emotion and Activity Recognition
import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import joblib

# Data manipulation and analysis
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    try:
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        from keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using sklearn models only")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(RANDOM_SEED)

print("🔧 INITIALIZING COMPLETE DATA PROCESSING SYSTEM")
print("=" * 60)

# Create project directory structure
output_dir = Path('/Users/sharvary_hh/Documents/6th Sem/Interdisciplinary EL/Project/model/output')
output_dir.mkdir(exist_ok=True)

# Create subdirectories
subdirs = ['data', 'logs', 'plots', 'reports', 'configs', 'models', 'weights']
for subdir in subdirs:
    (output_dir / subdir).mkdir(exist_ok=True)

print(f"📁 Project structure created at: {output_dir}")

# Configure logging
log_file = output_dir / 'logs' / f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ModelTraining')

# Enhanced System Configuration
class Config:
    """Enhanced configuration class"""
    
    def __init__(self):
        # Define feature columns
        self.FEATURES = [
            'Timestamp(ms)', 'ECG_Value', 'AccelX', 'AccelY', 'AccelZ',
            'GyroX', 'GyroY', 'GyroZ', 'TempC', 'IR', 'BPM', 'Avg_BPM'
        ]
        
        # Define all classes (updated list)
        self.ALL_CLASSES = [
            'Amusement', 'Awe', 'Enthusiasm', 'Liking', 'Surprised', 
            'Angry', 'Disgust', 'Fear', 'Sad', 
            'Standing', 'Sitting', 'Walking', 'Running'
        ]
        
        # Separate emotion and activity classes
        self.EMOTION_CLASSES = [
            'Amusement', 'Awe', 'Enthusiasm', 'Liking', 'Surprised', 
            'Angry', 'Disgust', 'Fear', 'Sad'
        ]
        
        self.ACTIVITY_CLASSES = [
            'Standing', 'Sitting', 'Walking', 'Running'
        ]
        
        # Model configuration
        self.SEQUENCE_LENGTH = 50
        self.TEST_SIZE = 0.2
        self.VAL_SIZE = 0.2
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.PATIENCE = 15

# Initialize configuration
config = Config()

# Enhanced patterns for new emotion classes
EMOTION_PATTERNS = {
    'Amusement': {'BPM_mult': 1.1, 'temp_offset': 0.3, 'ecg_var': 1.2},
    'Awe': {'BPM_mult': 1.05, 'temp_offset': 0.1, 'ecg_var': 0.9},
    'Enthusiasm': {'BPM_mult': 1.3, 'temp_offset': 0.6, 'ecg_var': 1.4},
    'Liking': {'BPM_mult': 1.05, 'temp_offset': 0.2, 'ecg_var': 1.0},
    'Surprised': {'BPM_mult': 1.3, 'temp_offset': 0.5, 'ecg_var': 1.5},
    'Angry': {'BPM_mult': 1.4, 'temp_offset': 0.8, 'ecg_var': 1.6},
    'Disgust': {'BPM_mult': 1.15, 'temp_offset': 0.3, 'ecg_var': 1.1},
    'Fear': {'BPM_mult': 1.5, 'temp_offset': 1.0, 'ecg_var': 1.8},
    'Sad': {'BPM_mult': 0.9, 'temp_offset': -0.2, 'ecg_var': 0.8}
}

# Activity patterns
ACTIVITY_PATTERNS = {
    'Standing': {'accel_var': 0.3, 'BPM_mult': 1.0},
    'Sitting': {'accel_var': 0.1, 'BPM_mult': 0.95},
    'Walking': {'accel_var': 1.5, 'BPM_mult': 1.15},
    'Running': {'accel_var': 3.0, 'BPM_mult': 1.6}
}

# Enhanced Feature Engineering Class
class AdvancedFeatureEngineering:
    """Advanced feature engineering for emotion and activity recognition"""
    
    def __init__(self):
        pass
    
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract time-domain features from signal"""
        features = {}
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal ** 2))
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        features['range'] = np.max(signal) - np.min(signal)
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['energy'] = np.sum(signal ** 2)
        return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray, fs: float = 100) -> Dict[str, float]:
        """Extract frequency-domain features"""
        features = {}
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        magnitude = np.abs(fft)
        
        # Spectral features
        features['spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]]
        features['spectral_flux'] = np.sum(np.diff(magnitude) ** 2)
        features['dominant_freq'] = freqs[np.argmax(magnitude[:len(magnitude)//2])]
        
        return features
    
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        features['median'] = np.median(signal)
        features['q25'] = np.percentile(signal, 25)
        features['q75'] = np.percentile(signal, 75)
        features['iqr'] = features['q75'] - features['q25']
        features['mad'] = np.median(np.abs(signal - np.median(signal)))
        features['entropy'] = stats.entropy(np.histogram(signal, bins=50)[0] + 1e-10)
        return features
    
    def extract_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive features from all signals"""
        feature_dict = {}
        
        # Process each sensor signal
        for col in ['ECG_Value', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'TempC', 'BPM']:
            if col in df.columns:
                signal = df[col].values
                
                # Time domain features
                time_features = self.extract_time_domain_features(signal)
                for feat_name, feat_val in time_features.items():
                    feature_dict[f'{col}_{feat_name}'] = feat_val
                
                # Frequency domain features
                freq_features = self.extract_frequency_domain_features(signal)
                for feat_name, feat_val in freq_features.items():
                    feature_dict[f'{col}_{feat_name}'] = feat_val
                
                # Statistical features
                stat_features = self.extract_statistical_features(signal)
                for feat_name, feat_val in stat_features.items():
                    feature_dict[f'{col}_{feat_name}'] = feat_val
        
        # Cross-sensor features
        if all(col in df.columns for col in ['AccelX', 'AccelY', 'AccelZ']):
            accel_magnitude = np.sqrt(df['AccelX']**2 + df['AccelY']**2 + df['AccelZ']**2)
            mag_features = self.extract_time_domain_features(accel_magnitude.values)
            for feat_name, feat_val in mag_features.items():
                feature_dict[f'accel_magnitude_{feat_name}'] = feat_val
        
        if all(col in df.columns for col in ['GyroX', 'GyroY', 'GyroZ']):
            gyro_magnitude = np.sqrt(df['GyroX']**2 + df['GyroY']**2 + df['GyroZ']**2)
            mag_features = self.extract_time_domain_features(gyro_magnitude.values)
            for feat_name, feat_val in mag_features.items():
                feature_dict[f'gyro_magnitude_{feat_name}'] = feat_val
        
        # HRV features
        if 'BPM' in df.columns:
            bpm_diff = np.diff(df['BPM'].values)
            if len(bpm_diff) > 0:
                feature_dict['hrv_rmssd'] = np.sqrt(np.mean(bpm_diff ** 2))
                feature_dict['hrv_sdnn'] = np.std(bpm_diff)
        
        # Convert to DataFrame
        result_df = df.copy()
        for feat_name, feat_val in feature_dict.items():
            result_df[feat_name] = feat_val
        
        return result_df

# Enhanced Data Generator
class EnhancedDataGenerator:
    """Generate realistic synthetic data for all emotion and activity classes"""
    
    def __init__(self, config):
        self.config = config
        
    def generate_dataset(self, n_samples: int = 2000) -> pd.DataFrame:
        """Generate synthetic dataset with all classes"""
        print(f"🔬 Generating enhanced synthetic dataset with {n_samples} samples...")
        
        # Initialize arrays
        data = {}
        for feature in self.config.FEATURES:
            data[feature] = np.zeros(n_samples)
        
        labels = []
        
        # Generate timestamps
        data['Timestamp(ms)'] = np.arange(n_samples) * 10
        time_seconds = data['Timestamp(ms)'] / 1000.0
        
        # Create segments for different classes
        segment_size = n_samples // len(self.config.ALL_CLASSES)
        
        for i, class_name in enumerate(self.config.ALL_CLASSES):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < len(self.config.ALL_CLASSES) - 1 else n_samples
            segment_length = end_idx - start_idx
            segment_time = time_seconds[start_idx:end_idx]
            
            # Get patterns
            if class_name in self.config.EMOTION_CLASSES:
                emotion_pattern = EMOTION_PATTERNS[class_name]
                activity_pattern = ACTIVITY_PATTERNS['Standing']  # Default activity
                pattern = {**emotion_pattern, **activity_pattern}
            else:
                emotion_pattern = {'BPM_mult': 1.0, 'temp_offset': 0.0, 'ecg_var': 1.0}
                activity_pattern = ACTIVITY_PATTERNS[class_name]
                pattern = {**emotion_pattern, **activity_pattern}
            
            # Generate ECG signal with emotion-specific variations
            ecg_base = 0.8 * np.sin(2 * np.pi * 1.2 * segment_time)
            ecg_var = pattern.get('ecg_var', 1.0)
            ecg_noise = np.random.normal(0, 0.1 * ecg_var, segment_length)
            data['ECG_Value'][start_idx:end_idx] = ecg_base + ecg_noise
            
            # Generate motion data based on activity
            if class_name == 'Walking':
                freq = 2.0
                data['AccelX'][start_idx:end_idx] = 2.0 * np.sin(2 * np.pi * freq * segment_time) + np.random.normal(0, 0.5, segment_length)
                data['AccelY'][start_idx:end_idx] = 1.5 * np.cos(2 * np.pi * freq * segment_time) + np.random.normal(0, 0.5, segment_length)
                data['AccelZ'][start_idx:end_idx] = 9.8 + 0.5 * np.sin(2 * np.pi * freq * segment_time) + np.random.normal(0, 0.3, segment_length)
            elif class_name == 'Running':
                freq = 3.5
                data['AccelX'][start_idx:end_idx] = 4.0 * np.sin(2 * np.pi * freq * segment_time) + np.random.normal(0, 1.0, segment_length)
                data['AccelY'][start_idx:end_idx] = 3.0 * np.cos(2 * np.pi * freq * segment_time) + np.random.normal(0, 1.0, segment_length)
                data['AccelZ'][start_idx:end_idx] = 9.8 + 1.0 * np.sin(2 * np.pi * freq * segment_time) + np.random.normal(0, 0.5, segment_length)
            else:
                # Standing, sitting, or emotional states
                accel_var = pattern.get('accel_var', 0.3)
                data['AccelX'][start_idx:end_idx] = np.random.normal(0, accel_var, segment_length)
                data['AccelY'][start_idx:end_idx] = np.random.normal(0, accel_var, segment_length)
                data['AccelZ'][start_idx:end_idx] = np.random.normal(9.8, accel_var * 0.5, segment_length)
            
            # Generate gyroscope data
            data['GyroX'][start_idx:end_idx] = np.gradient(data['AccelX'][start_idx:end_idx]) * 50 + np.random.normal(0, 50, segment_length)
            data['GyroY'][start_idx:end_idx] = np.gradient(data['AccelY'][start_idx:end_idx]) * 50 + np.random.normal(0, 50, segment_length)
            data['GyroZ'][start_idx:end_idx] = np.random.normal(0, 30, segment_length)
            
            # Generate temperature with emotion-specific variations
            temp_base = 37.0 + pattern.get('temp_offset', 0.0)
            data['TempC'][start_idx:end_idx] = temp_base + np.random.normal(0, 0.2, segment_length)
            
            # Generate BPM with both emotion and activity effects
            bpm_base = 75 * pattern.get('BPM_mult', 1.0)
            data['BPM'][start_idx:end_idx] = np.clip(bpm_base + np.random.normal(0, 5, segment_length), 50, 180)
            
            # Generate IR sensor data
            data['IR'][start_idx:end_idx] = np.random.randint(200, 800, segment_length)
            
            # Create labels
            labels.extend([class_name] * segment_length)
        
        # Calculate average BPM
        data['Avg_BPM'] = np.convolve(data['BPM'], np.ones(10)/10, mode='same')
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['target'] = labels
        
        print(f"✅ Enhanced dataset generated with shape: {df.shape}")
        print(f"Classes: {df['target'].nunique()}")
        return df

# Ensemble Model Class
class EmotionActivityEnsemble:
    """Ensemble model for emotion and activity recognition"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_selector = SelectKBest(f_classif, k=50)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.is_fitted = False
        
    def create_sklearn_models(self):
        """Create sklearn ensemble models"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=RANDOM_SEED
            ),
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=RANDOM_SEED
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=RANDOM_SEED
            )
        }
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft'
        )
        
        models['ensemble'] = voting_clf
        return models
    
    def create_deep_learning_model(self, input_shape):
        """Create deep learning model"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            Input(shape=input_shape),
            
            # CNN layers for feature extraction
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # LSTM layers for temporal modeling
            LSTM(128, return_sequences=True, dropout=0.3),
            LSTM(64, dropout=0.3),
            
            # Dense layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(self.config.ALL_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, df):
        """Prepare data for training or prediction"""
        # Extract comprehensive features
        df_features = self.feature_engineer.extract_comprehensive_features(df)
        
        # Separate features and target
        if 'target' in df_features.columns:
            X = df_features.drop(columns=['target'])
            y = df_features['target'].values
        else:
            X = df_features
            y = None
        
        # Remove timestamp column for modeling
        if 'Timestamp(ms)' in X.columns:
            X = X.drop(columns=['Timestamp(ms)'])
        
        return X, y
    
    def train(self, df):
        """Train the ensemble model"""
        print("🚀 Training ensemble model...")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
        )
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Train sklearn models
        self.models = self.create_sklearn_models()
        
        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'model': model
            }
            
            print(f"{name} - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        
        # Train deep learning model if available
        if TENSORFLOW_AVAILABLE:
            print("Training deep learning model...")
            
            # Prepare sequence data
            sequence_length = min(50, len(X_train_scaled))
            n_features = X_train_scaled.shape[1]
            
            # Create sequences
            X_train_seq = self.create_sequences(X_train_scaled, sequence_length)
            X_test_seq = self.create_sequences(X_test_scaled, sequence_length)
            
            # Adjust labels for sequences
            y_train_seq = y_train[sequence_length-1:]
            y_test_seq = y_test[sequence_length-1:]
            
            # One-hot encode labels
            y_train_cat = to_categorical(y_train_seq, num_classes=len(self.config.ALL_CLASSES))
            y_test_cat = to_categorical(y_test_seq, num_classes=len(self.config.ALL_CLASSES))
            
            # Create and train model
            dl_model = self.create_deep_learning_model((sequence_length, n_features))
            
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6),
                ModelCheckpoint(
                    str(output_dir / 'weights' / 'best_model.h5'),
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            history = dl_model.fit(
                X_train_seq, y_train_cat,
                validation_data=(X_test_seq, y_test_cat),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate deep learning model
            train_loss, train_acc = dl_model.evaluate(X_train_seq, y_train_cat, verbose=0)
            test_loss, test_acc = dl_model.evaluate(X_test_seq, y_test_cat, verbose=0)
            
            results['deep_learning'] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'model': dl_model,
                'history': history
            }
            
            print(f"Deep Learning - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
        
        self.is_fitted = True
        self.training_results = results
        
        # Save models
        self.save_models()
        
        return results
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for deep learning"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def predict(self, df):
        """Make predictions using ensemble"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        # Prepare data
        X, _ = self.prepare_data(df)
        
        # Remove timestamp
        if 'Timestamp(ms)' in X.columns:
            X = X.drop(columns=['Timestamp(ms)'])
        
        # Transform features
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model_info in self.training_results.items():
            if name == 'deep_learning':
                continue  # Handle separately
                
            model = model_info['model']
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            
            predictions[name] = pred
            probabilities[name] = prob
        
        # Ensemble prediction (majority vote with probability weighting)
        ensemble_probs = np.mean(list(probabilities.values()), axis=0)
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        
        # Decode predictions
        predicted_classes = self.label_encoder.inverse_transform(ensemble_pred)
        
        # Get confidence scores
        confidence_scores = np.max(ensemble_probs, axis=1)
        
        return predicted_classes, confidence_scores
    
    def save_models(self):
        """Save trained models"""
        model_dir = output_dir / 'models'
        
        # Save sklearn models
        for name, model_info in self.training_results.items():
            if name == 'deep_learning':
                continue
            
            model_path = model_dir / f'{name}_model.pkl'
            joblib.dump(model_info['model'], model_path)
        
        # Save preprocessing components
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, model_dir / 'label_encoder.pkl')
        joblib.dump(self.feature_selector, model_dir / 'feature_selector.pkl')
        
        print(f"💾 Models saved to {model_dir}")
    
    def load_models(self):
        """Load trained models"""
        model_dir = output_dir / 'models'
        
        # Load preprocessing components
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        self.label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
        self.feature_selector = joblib.load(model_dir / 'feature_selector.pkl')
        
        # Load sklearn models
        model_names = ['random_forest', 'gradient_boosting', 'svm', 'mlp', 'ensemble']
        self.training_results = {}
        
        for name in model_names:
            model_path = model_dir / f'{name}_model.pkl'
            if model_path.exists():
                model = joblib.load(model_path)
                self.training_results[name] = {'model': model}
        
        # Load deep learning model if available
        if TENSORFLOW_AVAILABLE:
            dl_model_path = output_dir / 'weights' / 'best_model.h5'
            if dl_model_path.exists():
                dl_model = tf.keras.models.load_model(dl_model_path)
                self.training_results['deep_learning'] = {'model': dl_model}
        
        self.is_fitted = True
        print("✅ Models loaded successfully")

# Main Training and Evaluation Function
def train_and_evaluate_model(csv_path=None):
    """Train and evaluate the complete model"""
    print("\n🎯 STARTING MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    # Initialize components
    generator = EnhancedDataGenerator(config)
    model = EmotionActivityEnsemble(config)
    
    # Load or generate data
    if csv_path and Path(csv_path).exists():
        print(f"📊 Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded data shape: {df.shape}")
    else:
        print("🔬 Generating synthetic training data...")
        df = generator.generate_dataset(3000)  # Larger dataset for training
        
        # Save generated data
        data_path = output_dir / 'data' / 'training_data.csv'
        df.to_csv(data_path, index=False)
        print(f"💾 Training data saved to {data_path}")
    
    # Display class distribution
    print(f"\n📊 Class Distribution:")
    class_counts = df['target'].value_counts()
    for class_name, count in class_counts.items():
        print(f"  • {class_name}: {count} samples")
    
    # Train model
    print(f"\n🚀 Training ensemble model...")
    training_results = model.train(df)
    
    # Display results
    print(f"\n📈 Training Results:")
    print("-" * 40)
    for name, results in training_results.items():
        print(f"{name.upper()}:")
        print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print()
    
    # Test prediction on sample data
    print(f"\n🔮 Testing Predictions:")
    print("-" * 40)
    
    # Create a small test sample
    test_df = generator.generate_dataset(100)
    predictions, confidence = model.predict(test_df)
    
    # Show some predictions
    for i in range(min(10, len(predictions))):
        actual = test_df.iloc[i]['target']
        predicted = predictions[i]
        conf = confidence[i]
        status = "✅" if actual == predicted else "❌"
        print(f"{status} Actual: {actual:12s} | Predicted: {predicted:12s} | Confidence: {conf:.3f}")
    
    # Calculate overall accuracy
    actual_labels = test_df['target'].values
    accuracy = np.mean(predictions == actual_labels)
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.4f}")
    
    return model, training_results

# Prediction Function for New Data
def predict_emotion_activity(model, csv_path):
    """Predict emotion/activity from new CSV data"""
    print(f"\n🔍 Making predictions for {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    
    # Make predictions
    predictions, confidence = model.predict(df)
    
    # Add predictions to dataframe
    df['predicted_class'] = predictions
    df['confidence'] = confidence
    
    # Save results
    results_path = output_dir / 'data' / 'predictions.csv'
    df.to_csv(results_path, index=False)
    
    print(f"💾 Predictions saved to {results_path}")
    
    # Show summary
    print(f"\n📊 Prediction Summary:")
    pred_counts = pd.Series(predictions).value_counts()
    for class_name, count in pred_counts.items():
        avg_conf = np.mean(confidence[predictions == class_name])
        print(f"  • {class_name}: {count} samples (avg confidence: {avg_conf:.3f})")
    
    return predictions, confidence

# Prediction Function for Manual Input
def predict_emotion_activity_manual(model, input_data):
    """
    Predict emotion/activity from manual input.
    input_data: dict or list of dicts with keys matching config.FEATURES
    """
    print("\n🔍 Making predictions for manual input")
    
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        raise ValueError("Input data must be a dict or list of dicts.")
    
    print(f"Input shape: {df.shape}")
    
    # Make predictions
    predictions, confidence = model.predict(df)
    
    # Add predictions to dataframe
    df['predicted_class'] = predictions
    df['confidence'] = confidence
    
    print("\n📊 Prediction Results:")
    for i, row in df.iterrows():
        print(f"  Row {i+1}: Predicted = {row['predicted_class']}, Confidence = {row['confidence']:.3f}")
    
    return predictions, confidence

# Main execution
if __name__ == "__main__":
    # Train the model
    model, results = train_and_evaluate_model()
    
    print("\n🎉 MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print("The model is ready for emotion and activity recognition.")
    print("To use the model on new data, call:")
    print("  predictions, confidence = predict_emotion_activity(model, 'your_data.csv')")
    print("\nRequired CSV columns:")
    print("  " + ", ".join(config.FEATURES))
    print("\nSupported classes:")
    print("  " + ", ".join(config.ALL_CLASSES))

    # --- Add this block for prediction on Data_log2.csv ---
    csv_path = "Data_log2.csv"
    if Path(csv_path).exists():
        print(f"\n🔍 Running prediction on {csv_path} ...")
        predictions, confidence = predict_emotion_activity(model, csv_path)
        print("\nSample predictions:")
        for i in range(min(10, len(predictions))):
            print(f"Row {i+1}: Predicted = {predictions[i]}, Confidence = {confidence[i]:.3f}")
    else:
        print(f"\n❌ File {csv_path} not found. Please check the path.")
    
    # --- Manual input prediction example ---
    manual_input = [
        {
            'Timestamp(ms)': 123456,
            'ECG_Value': 0.85,
            'AccelX': 0.1,
            'AccelY': 0.2,
            'AccelZ': 9.7,
            'GyroX': 5.0,
            'GyroY': 3.0,
            'GyroZ': 0.0,
            'TempC': 36.8,
            'IR': 400,
            'BPM': 80,
            'Avg_BPM': 78
        },
        {
            'Timestamp(ms)': 123466,
            'ECG_Value': 0.87,
            'AccelX': 0.12,
            'AccelY': 0.22,
            'AccelZ': 9.71,
            'GyroX': 5.1,
            'GyroY': 3.1,
            'GyroZ': 0.1,
            'TempC': 36.9,
            'IR': 410,
            'BPM': 81,
            'Avg_BPM': 78.5
        },
        # ...add at least 10-20 rows for meaningful feature extraction...
    ]
    print("\n🔍 Running prediction on manual input ...")
    predict_emotion_activity_manual(model, manual_input)
