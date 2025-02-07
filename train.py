import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from model import ContentModerationModel
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def train_model(train_texts, train_labels, val_texts, val_labels):
    print("\nTraining SVM model...")
    model = ContentModerationModel()

    # Train the model
    model.fit(train_texts, train_labels)

    # Evaluate on validation set
    val_probs, _ = model.predict(val_texts)
    val_preds = (val_probs >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(val_labels, val_preds)

    print("\nValidation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds))

    # Save the model and metrics
    os.makedirs('models/best_model', exist_ok=True)
    model.save('models/best_model')

    metrics = {
        'accuracy': accuracy,
        'threshold': model.threshold
    }

    with open('models/best_model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return model


def main():
    print("Loading data...")
    df = pd.read_csv('training_data.csv')

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(
        train_df['message'].tolist(),
        train_df['is_unsafe'].tolist(),
        val_df['message'].tolist(),
        val_df['is_unsafe'].tolist()
    )

    print("\nTraining completed!")

    # Optional: Test the model on a few examples
    test_texts = [
        "Hello, this is a test message",
        "Contact me at test@example.com",
        "Call me at 1234567890"
    ]

    probabilities, patterns = model.predict(test_texts)
    predictions = (probabilities >= model.threshold).astype(int)

    print("\nTest Predictions:")
    for text, prob, pred in zip(test_texts, probabilities, predictions):
        print(f"\nText: {text}")
        print(f"Probability of being unsafe: {prob:.4f}")
        print(f"Prediction: {'Unsafe' if pred else 'Safe'}")


if __name__ == "__main__":
    main()