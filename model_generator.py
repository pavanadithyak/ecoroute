"""
Script to generate a pre-trained RandomForest model for EcoRoutingAI
This script creates model.pkl with realistic training data for path selection decisions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_training_data(n_samples=2000):
    """
    Generate synthetic training data for the routing decision model.
    
    Features:
    - lat_latency: Latency of latency-optimal path (ms)
    - lat_carbon: Carbon cost of latency-optimal path (g CO₂)
    - eco_latency: Latency of eco-optimal path (ms) 
    - eco_carbon: Carbon cost of eco-optimal path (g CO₂)
    
    Target:
    - 0: Choose latency-optimal path
    - 1: Choose eco-optimal path
    """
    
    np.random.seed(42)
    
    data = []
    labels = []
    
    for _ in range(n_samples):
        # Generate base latency-optimal path metrics
        lat_latency = np.random.uniform(10, 100)
        lat_carbon = np.random.uniform(0.005, 0.08)
        
        # Eco path typically has higher latency but lower carbon
        eco_latency = lat_latency * np.random.uniform(1.05, 2.5)  # 5% to 150% higher
        eco_carbon = lat_carbon * np.random.uniform(0.2, 0.9)     # 10% to 80% of original
        
        # Calculate potential savings and penalties
        carbon_saving_pct = (lat_carbon - eco_carbon) / lat_carbon * 100
        latency_penalty_pct = (eco_latency - lat_latency) / lat_latency * 100
        
        # Decision logic based on realistic network optimization criteria
        # Choose eco path if:
        # 1. Carbon savings > 20% AND latency penalty < 60%
        # 2. OR carbon savings > 40% AND latency penalty < 100%
        # 3. OR carbon savings > 60% (regardless of latency for very green options)
        # 4. Add some randomness to simulate real-world uncertainty
        
        base_decision = 0
        
        if carbon_saving_pct > 60:
            base_decision = 1
        elif carbon_saving_pct > 40 and latency_penalty_pct < 100:
            base_decision = 1
        elif carbon_saving_pct > 20 and latency_penalty_pct < 60:
            base_decision = 1
        elif carbon_saving_pct > 30 and latency_penalty_pct < 30:
            base_decision = 1
        
        # Add 10% random noise to make model more realistic
        if np.random.random() < 0.1:
            base_decision = 1 - base_decision
        
        data.append([lat_latency, lat_carbon, eco_latency, eco_carbon])
        labels.append(base_decision)
    
    return np.array(data), np.array(labels)

def create_feature_names():
    """Return feature names for better model interpretability"""
    return [
        'latency_optimal_latency_ms',
        'latency_optimal_carbon_g',
        'eco_optimal_latency_ms', 
        'eco_optimal_carbon_g'
    ]

def main():
    """Generate and save the RandomForest model"""
    
    print("Generating training data...")
    X, y = generate_training_data(2000)
    feature_names = create_feature_names()
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Eco-path decisions: {sum(y_train)} / {len(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
    
    # Create and train the model
    print("\nTraining RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle any class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Latency-Optimal', 'Eco-Optimal']))
    
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.3f}")
    
    # Save the model
    print("\nSaving model to model.pkl...")
    joblib.dump(model, 'model.pkl')
    print("Model saved successfully!")
    
    # Test a few sample predictions
    print("\nSample Predictions:")
    sample_data = [
        [25.0, 0.02, 35.0, 0.01],  # Low latency penalty, good carbon savings
        [30.0, 0.05, 80.0, 0.02],  # High latency penalty, excellent carbon savings  
        [50.0, 0.03, 55.0, 0.025], # Small improvement in both metrics
    ]
    
    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)
    
    for i, (data, pred, prob) in enumerate(zip(sample_data, predictions, probabilities)):
        lat_lat, lat_carb, eco_lat, eco_carb = data
        carbon_save = (lat_carb - eco_carb) / lat_carb * 100
        latency_penalty = (eco_lat - lat_lat) / lat_lat * 100
        
        decision = "Eco-Optimal" if pred == 1 else "Latency-Optimal"
        confidence = max(prob) * 100
        
        print(f"Sample {i+1}: {decision} (confidence: {confidence:.1f}%)")
        print(f"  Carbon savings: {carbon_save:.1f}%, Latency penalty: {latency_penalty:.1f}%")

if __name__ == "__main__":
    main()
