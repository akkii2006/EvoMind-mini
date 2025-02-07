# EvoMind Mini

A lightweight version of the EvoMind content moderation system, using SVM-based classification for detecting and filtering messages containing personal information.

## About EvoMind Mini

EvoMind Mini is a simplified version of the full EvoMind content moderation system. While the main EvoMind (currently at version 1.2) uses advanced neural networks, EvoMind Mini employs a more lightweight Support Vector Machine (SVM) approach, making it suitable for smaller projects and demonstration purposes.

### Key Differences from Main EvoMind
- Uses SVM instead of Neural Networks
- Smaller model footprint
- Modified subset of the original EvoMind-1.0 dataset
- Optimized for smaller-scale applications
- Reduced computational requirements

## Features

- **Hybrid Detection Approach**:
  - Pattern-based detection for explicit contact information
  - SVM-based contextual understanding
  - Combined decision making

- **Detection Capabilities**:
  - Phone numbers in various formats
  - Email addresses
  - Requests for personal information
  - Basic context awareness

## Version Roadmap

EvoMind Mini follows the main EvoMind releases:
- Current: EvoMind Mini 1.0 (SVM-based)
- Future: When EvoMind 1.3 launches, EvoMind Mini 1.1 will follow
- Upcoming versions will transition to Neural Network architecture

## Requirements

- Python 3.8 or higher
- See requirements.txt for full dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akkii2006/EvoMind-mini.git
cd EvoMind-mini
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Mac/Linux:
```bash
source venv/bin/activate
```

4. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Using Pre-trained Model
The repository comes with a pre-trained model. To use it:
```bash
python test_moderator.py
```

### Training Custom Model
While the original training dataset is not included, you can create and train on your own dataset:

1. Modify `create_dataset.py` to use your custom data
2. Generate the dataset:
```bash
python create_dataset.py
```

3. Train the model:
```bash
python train.py
```

## Model Architecture

The current version uses:
1. **Pattern Detector**:
   - Regular expression based detection
   - Common pattern matching
   - Direct feature extraction

2. **SVM Classifier**:
   - TF-IDF feature vectorization
   - RBF kernel
   - Probability calibration

3. **Combined Decision Making**:
   - Pattern detection results
   - SVM classification scores
   - Threshold-based final decision

## Important Notes

- The model performs best with clear, structured text input
- CPU-only implementation - no GPU requirements
- Supports batch processing for multiple messages
- Easy integration via the ContentModerator class
- Model thresholds can be configured through metrics.json

## Performance Metrics

The SVM-based model achieves:
- Baseline accuracy: ~85%
- Edge case handling: ~75%
- Context awareness: Limited compared to full version
- Fast inference time: <50ms on CPU

## Contributing

1. Fork the repository
2. Create your feature branch
3. Create a pull request

## Contact

For the full EvoMind solution or enterprise needs, please contact Rujevo AI.

## License

MIT License - See LICENSE file for details


## Made with ❤️ by Rujevo AI
