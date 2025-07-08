# 🚀 Transformer from Scratch using PyTorch

- This repository contains a full implementation of the Transformer model built from scratch using **PyTorch**, inspired by the paper _[“Attention Is All You Need”](https://arxiv.org/abs/1706.03762)_ by Vaswani et al. It includes both the **encoder** and **decoder** architecture without relying on libraries like Hugging Face.

---

## ✨ Features

- ✅ Full Transformer Architecture (Encoder + Decoder)
- ✅ Multi-Head Self-Attention Mechanism
- ✅ Custom Sinusoidal Positional Encoding
- ✅ Layer Normalization, Residual Connections, Dropout
- ✅ Custom Dataset Loader with Tokenization and Masking
- ✅ Training-ready pipeline for NLP sequence tasks

---

🧪 Dataset
You can use any bilingual translation dataset, such as from HuggingFace's datasets library (e.g., "opus_books"). The dataset loader handles:

- Source & Target language tokenization

- Special tokens: [SOS], [EOS], [PAD]

- Fixed-length padding and causal masks

---

🧠 Components Built from Scratch
- InputEmbeddings

- PositionalEncoding

- MultiHeadAttentionBlock

- FeedForwardBlock

- ResidualConnection

- EncoderBlock, DecoderBlock

- Transformer

- BilingualDataset



## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Harikrushna2272/Transformer_Implementation.git
cd Transformer_Implementation
pip install -r requirements.txt

