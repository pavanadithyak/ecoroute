import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

def generate_training_data(n_samples=2000):
    """Generate synthetic training data for routing decisions"""
    np.random.seed(42)
    
    data = []
    labels = []
    
    for _ in range(n_samples):
        lat_latency = np.random.uniform(10, 100)
        lat_carbon = np.random.uniform(0.005, 0.08)
        
        eco_latency = lat_latency * np.random.uniform(1.05, 2.5)
        eco_carbon = lat_carbon * np.random.uniform(0.2, 0.9)
        
        carbon_saving_pct = (lat_carbon - eco_carbon) / lat_carbon * 100
        latency_penalty_pct = (eco_latency - lat_latency) / lat_latency * 100
        
        base_decision = 0
        
        if carbon_saving_pct > 60:
            base_decision = 1
        elif carbon_saving_pct > 40 and latency_penalty_pct < 100:
            base_decision = 1
        elif carbon_saving_pct > 20 and latency_penalty_pct < 60:
            base_decision = 1
        elif carbon_saving_pct > 30 and latency_penalty_pct < 30:
            base_decision = 1
        
        if np.random.random() < 0.1:
            base_decision = 1 - base_decision
        
        data.append([lat_latency, lat_carbon, eco_latency, eco_carbon])
        labels.append(base_decision)
    
    return np.array(data), np.array(labels)

def create_models():
    """Create and train multiple AI models for comparison"""
    X, y = generate_training_data(2000)
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),
        'NeuralNetwork': MLPClassifier(
            hidden_layer_sizes=(50, 30),
            max_iter=1000,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
        joblib.dump(model, f'model_{name.lower()}.pkl')
    
    return trained_models

def load_all_models():
    """Load all available models"""
    model_names = ['randomforest', 'gradientboosting', 'decisiontree', 'neuralnetwork']
    models = {}
    
    for name in model_names:
        try:
            models[name] = joblib.load(f'model_{name}.pkl')
        except FileNotFoundError:
            pass
    
    if not models:
        trained = create_models()
        models = {k.lower(): v for k, v in trained.items()}
    
    return models

def compare_models(features):
    """Compare predictions from all models"""
    models = load_all_models()
    results = {}
    
    for name, model in models.items():
        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
        results[name] = {
            'prediction': prediction,
            'confidence': max(proba) * 100,
            'eco_probability': proba[1] * 100
        }
    
    return results
