# 🏢 Hepsiburada Address Classification with XLM-RoBERTa + LoRA

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏆 Competition Context

This project was developed for the **[Hepsiburada Hackathon on Kaggle](https://www.kaggle.com/competitions/hepsiburada-hackathon-kaggle-etabi/overview)**, where I:
- **Ranked 20th** on the public leaderboard among data scientists and ML engineers
- Achieved **67.7% F1-Score** on a multi-class problem with 10,000+ classes

## 🎯 Problem Statement

Hepsiburada, one of Turkey's largest e-commerce platforms, needed an automated system to classify delivery addresses into specific area labels to optimize their logistics operations. The challenge involved predicting over **10,000 unique area labels** from unstructured Turkish address text.

### Key Challenges:
- **Extreme Class Imbalance:** 10,000+ unique labels with varying frequencies
- **Unstructured & Inconsistent Text:** Different address formats (e.g., "mah." vs "mahalle" vs "mahallesi"), abbreviations, and noise needed to be standardized into a single consistent format that the model could understand
- **High Spelling Error Rate:** Significant typos and misspellings (such as "Kocarli", "Cumhuriyey caddesi", "iz mir") in raw address data requiring robust normalization
- **Computational Constraints:** Fine-tuning large models efficiently with limited resources
- **Limited Turkish NLP Resources:** Scarcity of high-quality pre-trained models specifically optimized for Turkish language on Hugging Face, requiring careful model selection

## 📊 Dataset

- **Total Samples:** 848,080 Turkish addresses
- **Task:** Multi-class classification
- **Classes:** 10,000+ unique area labels
- **Source:** Hepsiburada Hackathon (Kaggle)
- **Format:** Structured addresses paired with area labels

**Note:** The full dataset is proprietary and not included in this repository. A sample of 50 addresses is provided in [`data/sample_data.csv`](data/sample_data.csv) for reference.

### Example Data:
```csv
label,structured_address
8831,akarca mahallesi adnan menderes caddesi 864 sokak
3067,ismet inönü mahallesi 2001 sokak çeşme belediyesi çeşme
8210,gazeteci hasan tahsin caddesi
```

## Solution Overview

### Model Architecture
I implemented a **transformer-based classification pipeline** using XLM-RoBERTa-Large fine-tuned with LoRA:

```
Raw Address → Text Preprocessing → XLM-RoBERTa-Large + LoRA → Label (10K+ classes)
```

### Why XLM-RoBERTa + LoRA?

I experimented with multiple transformer architectures before selecting the final model:
- **XLM-RoBERTa-Base:** Faster but lower accuracy (~62% F1)
- **DistilBERT:** Lightweight but struggled with Turkish morphology (~58% F1)
- **XLM-RoBERTa-Large:** Best performance-resource trade-off ✅

**Final Choice - XLM-RoBERTa-Large:**
   - Pre-trained on 100+ languages including Turkish
   - 550M parameters with strong multilingual understanding
   - Excellent at capturing semantic meaning from text
   - Achieved 67.7% F1-Score (best among tested models)

**LoRA (Low-Rank Adaptation):**
   - Parameter-efficient fine-tuning technique
   - Trained only **~1M parameters** instead of 550M
   - Reduced memory usage by 3x, faster training
   - Essential for working with limited computational resources

##  Technical Implementation

### 1. Data Preprocessing Pipeline ([`preprocessing.py`](preprocessing.py))

I developed a Turkish text preprocessing pipeline:

**Key Steps:**
```python
# Province/District Normalization
- Used tr_kktc_postal_codes.csv to standardize location names
- Fixed spacing issues: "karahayıt" → "kara hayıt"

# Noise Removal
- Removed apartment numbers, floor info, door numbers
- Cleaned patterns: "No:17/5 Daire:8" → removed

# Text Normalization
- Lowercase conversion
- Punctuation removal
- Whitespace cleanup
- Directional indicator removal (karşısı, yanı, etc.)
```

**Example Transformation:**
```
Input:  "Atatürk Mah. 123 Sok. No:17/5 Daire:8 Konak İzmir"
Output: "atatürk mahallesi 123 sokak konak izmir"
```

**Impact:** Clean, normalized addresses improved model convergence and reduced noise-related errors.

### 2. Model Training Pipeline ([`model_training.py`](model_training.py))

#### Architecture Details:
- **Base Model:** `xlm-roberta-large` (560M parameters)
- **LoRA Configuration:**
  ```python
  LoraConfig(
      r=16,                   
      lora_alpha=32,           
      target_modules=["query", "value"],  
      lora_dropout=0.05,
      bias="none"
  )
  ```

#### Training Strategy:

**Hyperparameters:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Batch Size | 256 (effective) | Gradient accumulation (2 steps × 128) |
| Learning Rate | 5e-5 | Conservative for stable convergence |
| Warmup Steps | 10% | Gradual LR increase prevents early instability |
| Epochs | 25 (max) | Early stopping with patience=2 |
| Label Smoothing | 0.05 | Prevents overconfidence on imbalanced classes |

**Optimization Techniques:**
1. **Mixed Precision Training (FP16):**
   - 2x faster training, 50% memory reduction
   - No loss in model quality

2. **Gradient Accumulation:**
   - Simulated large batch size (256) on limited GPU memory
   - Improved gradient stability

3. **Learning Rate Scheduling:**
   - Linear warmup (10% steps)
   - Linear decay to prevent overfitting

4. **Data Augmentation:**
   - 2x training data through intelligent duplication
   - Helped model generalize on rare classes

5. **Early Stopping:**
   - Monitored validation F1-score
   - Stopped when no improvement for 2 epochs
   - Prevented overfitting on training set

#### Training Infrastructure:
- **Hardware:** NVIDIA A100 GPU (40GB VRAM)
- **Training Time:** ~9 hours for 25 epochs
- **Framework:** PyTorch 2.0 with Hugging Face Transformers

## 📈 Results & Performance

### Final Metrics:
| Metric | Score |
|--------|-------|
| **Validation F1-Score (Macro)** | **67.7%** |
| **Validation Accuracy** | 68.2% |
| **Training Time** | 9 hours (A100) |
| **Trainable Parameters** | ~1M (0.18% of total) |
| **Total Parameters** | 560M (XLM-RoBERTa-Large) |

### Performance Analysis:

**What Worked Well:**
- ✅ LoRA achieved comparable performance to full fine-tuning with 99% fewer trainable parameters
- ✅ Text preprocessing significantly improved model's ability to learn location patterns
- ✅ XLM-RoBERTa's multilingual pre-training captured Turkish linguistic nuances effectively
- ✅ Label smoothing helped model perform better on imbalanced classes

**Challenges:**
- ⚠️ Rare classes (< 10 samples) were difficult to predict accurately
- ⚠️ Inconsistent address formatting (e.g., "mah." vs "mahalle" vs "mahallesi") required standardization
- ⚠️ High spelling error rate in raw data needed careful preprocessing and normalization


## 🎓 Key Takeaways & Skills Demonstrated

### Technical Skills:
- ✅ **NLP & Transformers:** Fine-tuning large language models (XLM-RoBERTa) for text classification
- ✅ **Parameter-Efficient Fine-Tuning:** Implementing LoRA for resource-constrained training
- ✅ **Turkish Language Processing:** Handling low-resource language challenges with limited pre-trained model availability
- ✅ **Deep Learning Optimization:** Mixed precision, gradient accumulation, learning rate scheduling
- ✅ **Data Engineering:** Text preprocessing, normalization, augmentation strategies
- ✅ **Production ML:** Early stopping, validation strategies, model evaluation

### Project Management:
- ✅ End-to-end ML pipeline development (preprocessing → training → evaluation)
- ✅ Working with large-scale datasets (848K samples)
- ✅ Computational resource optimization (LoRA for efficiency)
- ✅ Competition-focused: Achieved 67.7% F1 in Kaggle hackathon environment

## 📁 Project Structure

```
hepsiburada_address_classification/
├── README.md                      # Project documentation (this file)
├── requirements.txt               # Python dependencies
├── preprocessing.py               # Turkish address cleaning & normalization
├── model_training.py             # XLM-RoBERTa + LoRA training pipeline
├── data/
│   ├── sample_data.csv           # 50 example addresses
│   └── tr_kktc_postal_codes.csv  # Turkish postal codes reference
└── LICENSE                        # MIT License
```

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch 2.0, Hugging Face Transformers, PEFT (LoRA)
- **NLP:** XLM-RoBERTa-Large, Tokenization, Text Classification
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Optimization:** Mixed Precision (AMP), Gradient Accumulation, AdamW
- **Infrastructure:** NVIDIA A100 GPU, CUDA

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hepsiburada** for organizing the hackathon and providing the dataset
- **Hugging Face** for the Transformers library and model hub
- **Microsoft Research** for developing LoRA (Low-Rank Adaptation)

## 📧 Contact

**Yusuf** - [GitHub](https://github.com/yusuf3662)

---

⭐ **If you found this project interesting, please give it a star!** ⭐
