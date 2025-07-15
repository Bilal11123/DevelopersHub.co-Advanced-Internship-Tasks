# News Topic Classification using BERT

## Objective
This project fine-tunes a BERT model to classify news headlines from the AG News dataset into four categories: World, Sports, Business, and Sci/Tech. The implementation focuses on achieving high accuracy while providing a complete pipeline from training to deployment.

## Methodology

### Implementation Details
1. **Dataset**: 
   - AG News dataset (120,000 training samples, 7,600 test samples)
   - Automatically loaded via Hugging Face's `datasets` library

2. **Model Architecture**:
   - Base model: `bert-base-uncased`
   - Added classification head with 4 output units
   - Default BERT weights initialized from pre-trained model

3. **Training Configuration**:
   - Batch size: 16 (per device)
   - Learning rate: 2e-5
   - Training epochs: 3
   - Weight decay: 0.01
   - Maximum sequence length: 128 tokens
   - Padding strategy: Fixed length padding

4. **Evaluation Metrics**:
   - Accuracy
   - Weighted F1-score

### Key Components
- **Tokenization**: Uses `BertTokenizerFast` for efficient processing
- **Training Loop**: Leverages Hugging Face's `Trainer` API
- **Model Saving**: Saves both model and tokenizer for deployment

## Results

### Performance Metrics
| Metric        | Value   |
|---------------|---------|
| Training Time | ~2 hours|
| Accuracy      | 94.8%   |
| F1-score      | 94.8%   |

### Training Observations
1. The model achieves excellent classification performance out of the box
2. Fixed-length padding simplifies implementation but may be less memory-efficient
3. Default hyperparameters work well for this task without extensive tuning
4. Three epochs proved sufficient for convergence

### Project Structure
Task 1/
1. ├── train.py             # Training script
2. ├── app.py               # Streamlit application
3. ├── ag_news_bert/        # Saved model directory
4. ├── README.md            # This file
5. └── requirements.txt     # Python dependencies

## Installation & Usage

### Requirements
```bash
pip install transformers datasets torch scikit-learn
pip install -U datasets
pip install fsspec==2023.9.2
