# **Attention Is All You Need - Language Translation (English to Italian)** 🚀

## **📌 Project Overview**
This project is an implementation of the famous **Transformer model** from the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) using **PyTorch**. The model is trained to perform **language translation** from **English to Italian**, demonstrating the power of **self-attention and positional encoding** for sequence-to-sequence learning.

Unlike traditional RNN-based approaches, the Transformer uses **self-attention mechanisms** and a **fully parallelizable architecture**, making it highly efficient for **natural language processing (NLP) tasks** such as machine translation.

---

## **🧠 Model Architecture: Transformer**
The **Transformer** model consists of:
1. **Encoder**: Converts input text into contextual embeddings using multi-head self-attention.
2. **Decoder**: Generates translated output while attending to encoder embeddings.
3. **Multi-Head Self-Attention**: Helps the model focus on different parts of the sentence.
4. **Positional Encoding**: Injects sequence order information since Transformers have no recurrence.

### **📌 Why Use Transformers for Translation?**
✅ **Fully Parallelizable**: Unlike RNNs, it processes entire sequences at once.  
✅ **Better Context Understanding**: Self-attention allows the model to relate words across long distances.  
✅ **Scalability**: Works well on large datasets and modern hardware (GPUs/TPUs).  
✅ **State-of-the-Art Performance**: Used in models like BERT, GPT, and T5.

---

## **📂 Dataset**
For training, we use the **English-Italian dataset** from **Tatoeba**:
- **Source Language:** English
- **Target Language:** Italian
- **Dataset Size:** 100K+ sentence pairs

### **📥 Download Dataset**
You can download the dataset using:
```bash
wget https://raw.githubusercontent.com/facebookresearch/LASER/master/data/tatoeba/v1/tatoeba.eng-ita.tsv
```

---

## **🚀 Steps to Run the Project**

### **1️⃣ Install Dependencies**
Ensure you have Python **3.7+** and install the required libraries:
```bash
pip install torch torchvision torchaudio numpy tqdm spacy
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
```

### **2️⃣ Preprocess the Data**
Tokenize the dataset and convert it into **PyTorch tensors**:
```python
from preprocess import prepare_data
train_loader, val_loader, tokenizer_en, tokenizer_it = prepare_data()
```

### **3️⃣ Train the Transformer Model**
Run the training script:
```bash
python train.py
```

This will train the Transformer model from scratch using the **Adam optimizer** and **cross-entropy loss**.

### **4️⃣ Perform Translation**
Once trained, test the model with:
```python
from translate import translate
print(translate("How are you?"))  # Expected Output: "Come stai?"
```

---

## **📌 Features & Optimizations**
✅ **Implemented Transformer from Scratch** using PyTorch.  
✅ **Uses Byte Pair Encoding (BPE)** for efficient subword tokenization.  
✅ **Multi-GPU Support** for faster training.  
✅ **Attention Visualization** for understanding model behavior.  
✅ **Beam Search Decoding** for better translation accuracy.  

---

## **📌 Future Improvements**
🔹 **Use a Pretrained Transformer Model (e.g., mBART, MarianMT)** for better results.  
🔹 **Expand to Multiple Languages (e.g., French, German)** for multilingual support.  
🔹 **Optimize Training with Mixed Precision (FP16)** for faster execution.

---

## **👨‍💻 Contributors**
- **Hammad Khan** *(Lead Developer)*
- Inspired by **Vaswani et al. (2017)** *(Attention Is All You Need)*

---

## **📜 License**
This project is open-source and follows the **MIT License**.

🚀 **Happy Coding & Happy Translating!** 🌍

