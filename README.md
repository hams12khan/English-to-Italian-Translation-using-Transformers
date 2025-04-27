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
