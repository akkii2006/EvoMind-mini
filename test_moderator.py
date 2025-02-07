import os
from model import ContentModerationModel
import numpy as np


class ContentModerator:
    def __init__(self, model_path='models/best_model'):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found! Please train the model first.")

        self.model = ContentModerationModel.load(model_path)

        # Load metrics if available
        metrics_path = os.path.join(model_path, 'metrics.json')
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.model.threshold = metrics.get('threshold', 0.75)

    def check_message(self, message):
        # Get prediction and patterns
        probabilities, patterns = self.model.predict([message])
        unsafe_score = probabilities[0]  # Get first prediction since we only have one message

        # Check for patterns
        has_patterns = bool(patterns.get('phone_numbers', []) or patterns.get('emails', []))
        final_score = 0.9 if has_patterns else unsafe_score

        return {
            'is_unsafe': has_patterns or final_score >= self.model.threshold,
            'unsafe_score': float(final_score),  # Convert numpy float to Python float
            'safe_score': float(1 - final_score),
            'detected_patterns': patterns
        }


def main():
    print("\nInitializing Content Moderator...")
    moderator = ContentModerator()

    while True:
        message = input("\n" + "=" * 50 + "\nEnter message (or 'quit'): ").strip()
        if not message or message.lower() == 'quit':
            break

        try:
            result = moderator.check_message(message)
            print(f"\nRESULTS:\n{'-' * 50}")
            print(f"Message: {message}")
            print(f"Status: {'❌ UNSAFE' if result['is_unsafe'] else '✅ SAFE'}")
            print(f"Safe Score: {result['safe_score']:.2%}")
            print(f"Unsafe Score: {result['unsafe_score']:.2%}")

            if result['detected_patterns']:
                print("\nDetected Patterns:")
                for type_, matches in result['detected_patterns'].items():
                    if matches:
                        print(f"{type_}: {matches}")

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        print("\nThank you for using the Content Moderator!")