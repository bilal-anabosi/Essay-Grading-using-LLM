# Automatic Essay Grading using Instruction-Tuned Transformers

## Overview

This project presents an automated essay grading system that utilizes a 4-bit quantized instruction-tuned transformer model (Mistral-7B-Instruct-v0.2) to assess student essays. The system provides both numerical scores (0-4) and detailed rationales, simulating expert grading behavior with high consistency, this was done by using loRA and instruction tuning.

## Team Members

- **Bilal Anabosi**
- **Toqa Asedah** 
- **Ahmad Istieteh**

**Supervisor:** Dr. Hamed Abdelhaq  
**Institution:** An-Najah National University, College of Engineering and Information Technology

## Key Features

- 🎯 **Accurate Scoring**: Achieves 72.96% exact match accuracy and 99.15% within-1 accuracy
- 📊 **Strong Correlation**: Pearson correlation of 0.91 with human graders
- 🔍 **Interpretable Results**: Generates detailed rationales with 0.76 semantic similarity to human explanations
- ⚡ **Efficient Training**: Uses LoRA and 4-bit quantization for resource-efficient fine-tuning


## System Architecture

### Workflow

![Workflow (Community)](https://github.com/user-attachments/assets/67c87781-8e1d-4274-922d-1b91c4412c83)



The system follows this process:
1. **Input Processing**: Takes essay question, reference answer, student answer, and mark scheme
2. **Instruction Formatting**: Converts inputs into structured prompts for the model
3. **Model Inference**: Fine-tuned Mistral-7B processes the prompt
4. **Output Generation**: Produces score (0-4) and detailed rationale
5. **Evaluation**: Compares against expert grades and rationales

### Model Details

- **Base Model**: Mistral-7B-Instruct-v0.2 (4-bit quantized)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with optimal parameters
- **Training Environment**: Google Colab with Tesla T4 GPUÂ
- **Quantization**: 4-bit (nf4) with float16 compute

## Dataset

- **Total Samples**: 3,548 AI-generated essay examples
- **Split**: 80% training (2,838), 10% validation (355), 10% test (355)
- **Format**: JSONL with structured fields
- **Components**:
  - Essay question
  - Expert reference answer
  - Student answer
  - Mark scheme (0-4 points)
  - Ground truth score and rationale

## Training Configuration

### LoRA Parameters (Optimized)
```python
r=32                    # Rank
lora_alpha=32          # Alpha scaling
lora_dropout=0.0       # Dropout rate
target_modules=[       # Targeted layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Setup
- **Epochs**: 1
- **Batch Size**: 16 (effective)
- **Optimizer**: AdamW (8-bit)
- **Learning Rate**: 2e-4
- **Training Loss**: ~0.28
- **Validation Loss**: ~0.34

## Results

### Performance Metrics

<img width="649" height="547" alt="Unknown-3" src="https://github.com/user-attachments/assets/0ce46307-bfc5-4fa8-b103-187466f9e37e" />


#### Score Prediction Accuracy
- **Mean Absolute Error (MAE)**: 0.26
- **Root Mean Square Error (RMSE)**: 0.5281
- **Pearson Correlation**: 0.9179
- **Exact Match Accuracy**: 75.96%
- **Within-1 Accuracy**: 99.15%
- **Perplexity**: 1.42

#### F1 Scores by Class
| Score | F1 Score |
|-------|----------|
| 0     | 0.79     |
| 1     | 0.72     |
| 2     | 0.58     |
| 3     | 0.82     |
| 4     | 0.66     |

### Rationale Quality

<img width="686" height="470" alt="Unknown-2" src="https://github.com/user-attachments/assets/b3a1f172-a076-4ff9-84fb-e15221fc4c63" />


- **Average Cosine Similarity**: 0.7561
- Indicates model-generated rationales are semantically close to human-written explanations
- Provides interpretable feedback for educational use

## Installation & Usage

### Prerequisites
```bash
pip install torch transformers unsloth accelerate
```

### Model Loading
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### Inference Example
```python
def grade_essay(question, reference_answer, student_answer, mark_scheme):
    prompt = f"""
    Instructions:
    - Grade the student answer on a scale from 0 to 4 based strictly on the mark scheme.
    - For each criterion, assess whether it was satisfied.
    - Provide a detailed and objective rationale explaining the score.
    - Be concise, specific, and professional in your explanation.
    
    Question: {question}
    Reference Answer: {reference_answer}
    Student Answer: {student_answer}
    Mark Scheme: {mark_scheme}
    """
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, temperature=0.1, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Key Findings

1. **Optimal LoRA Configuration**: r=32, alpha=32 provided the best balance between performance and efficiency
2. **High Reliability**: 99.15% within-1 accuracy ensures practical applicability
3. **Semantic Coherence**: 0.76 cosine similarity demonstrates meaningful rationale generation
4. **Efficient Training**: 4-bit quantization enabled training on limited computational resources
5. **Scalability**: System design supports large-scale educational assessment

## Challenges & Solutions

### Overfitting Prevention
- Higher LoRA ranks (r=64) showed signs of overfitting
- Optimal configuration found through systematic experimentation
- Balanced performance and generalization

### Resource Constraints
- 4-bit quantization reduced memory requirements
- LoRA adaptation enabled efficient fine-tuning
- Single epoch training prevented overfitting on small dataset

## Future Work

- Expand dataset with more diverse essay types
- Investigate multi-lingual grading capabilities
- Implement active learning for continuous improvement
- Deploy as web service for educational institutions
- Explore integration with learning management systems
