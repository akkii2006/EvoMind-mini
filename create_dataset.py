"""
EvoMind Mini - Dataset Generator
------------------------------------------------
To add your own data:
1. Add safe messages to safe_templates
2. Add unsafe message patterns to unsafe_templates
3. Adjust the generate_dataset parameters as needed

Made by Mistyoz AI
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
from tqdm import tqdm


class MessageGenerator:
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.load_templates()

    def load_templates(self):
        """
        Load message templates

        # ADD YOUR CUSTOM DATA HERE:
        - Add to safe_templates for allowed messages
        - Add to unsafe_templates for messages to detect
        """

        # Safe message templates
        self.safe_templates = [
            # Business Communications
            "The meeting starts at {time}",
            "Project update: {status}",
            "Please review the document",
            "Sprint planning at {time}",

            # General Messages
            "Thank you for your help",
            "How are you today?",
            "Great work on the project",
            "Looking forward to the meeting",

            # ADD YOUR SAFE MESSAGES HERE
            # Example:
            # "What is the progress like?",
        ]

        # Unsafe message templates (requesting contact info)
        self.unsafe_templates = [
            # Basic Patterns
            "What's your number?",
            "Can you share your contact?",
            "Let's connect outside",
            "Send me your details",

            # Professional-sounding
            "Could we discuss this offline?",
            "Can we connect directly?",

            # ADD YOUR UNSAFE PATTERNS HERE
            # Example:
            # "Send me your email.",
        ]

        # Message variables
        # Add more variables for more varied dataset generation.
        self.variables = {
            'time': ['2 PM', '3:30 PM', '10 AM', '4:15 PM'],
            'status': ['in progress', 'completed', 'under review', 'starting soon']
        }

    def fill_template(self, template: str) -> str:
        """Fill template with random variables"""
        message = template
        for var, values in self.variables.items():
            if '{' + var + '}' in message:
                message = message.replace('{' + var + '}', random.choice(values))
        return message

    def generate_messages(self, templates: List[str], count: int) -> List[str]:
        """Generate messages from templates"""
        messages = []
        while len(messages) < count:
            template = random.choice(templates)
            message = self.fill_template(template)
            messages.append(message)
        return messages

    def create_dataset(
            self,
            output_file: str = 'training_data.csv',
            safe_count: int = 100,  # Adjust these numbers as needed
            unsafe_count: int = 50
    ) -> pd.DataFrame:
        """Create the training dataset"""
        print("\nGenerating dataset...")

        # Generate messages
        safe_messages = self.generate_messages(self.safe_templates, safe_count)
        unsafe_messages = self.generate_messages(self.unsafe_templates, unsafe_count)

        # Combine and label messages
        messages = safe_messages + unsafe_messages
        labels = [0] * len(safe_messages) + [1] * len(unsafe_messages)

        # Create DataFrame
        df = pd.DataFrame({
            'message': messages,
            'is_unsafe': labels
        })

        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to CSV
        df.to_csv(output_file, index=False)

        # Print summary
        print("\nDataset Summary:")
        print(f"Total messages: {len(df)}")
        print(f"Safe messages: {len(safe_messages)} ({len(safe_messages) / len(df) * 100:.1f}%)")
        print(f"Unsafe messages: {len(unsafe_messages)} ({len(unsafe_messages) / len(df) * 100:.1f}%)")

        return df


def main():
    """Generate the dataset"""
    try:
        print("Initializing Message Generator...")
        generator = MessageGenerator()

        # Generate dataset
        # CUSTOMIZE THESE PARAMETERS AS NEEDED:
        df = generator.create_dataset(
            output_file='training_data.csv',
            safe_count=100,  # Number of safe messages to generate
            unsafe_count=50  # Number of unsafe messages to generate
        )

        print("\nDataset created successfully!")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
