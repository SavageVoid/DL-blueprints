# Deep Learning Architectures and Reference Implementations

This repository contains clean, end-to-end reference implementations of key neural network architectures in NumPy, PyTorch, and TensorFlow/Keras. The project covers foundational deep learning, computer vision, sequence modeling, transformer visualization, generative models, and graph neural networks.

## Built With

Python, PyTorch, TensorFlow, Keras, Hugging Face, PyTorch Geometric, NumPy

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
</p>

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Core Language | Python 3.10+ | Primary development language |
| Deep Learning | PyTorch | Model implementation for VAE, GAN, GCN, and BERT fine-tuning |
| Deep Learning | TensorFlow / Keras | Model implementation for CIFAR-10, MNIST, LSTM, and MRI brain tumor detection |
| Transformer Models | Hugging Face Transformers | Fine-tuning and self-attention weight extraction (`bert-base-uncased`) |
| Graph Deep Learning | PyTorch Geometric (PyG) | Node classification GCN models on the Cora dataset |
| Math & Data Processing | NumPy & Pandas | Feed-forward neural network from scratch, data pipeline manipulation |
| Visualization | Matplotlib & Seaborn | Attention heatmaps, training loss/accuracy curves, and model predictions |

---

## Directory Structure

```
.
├── fundamentals/
│   ├── pytorch_vs_tensorflow_basics.ipynb
│   └── xor_backpropagation_from_scratch.ipynb
│
├── computer_vision/
│   ├── cifar10_cnn_classifier.ipynb
│   ├── mnist_digit_classifier.ipynb
│   └── brain_tumor_mri_detection.ipynb
│
├── nlp/
│   ├── twitter_sentiment_lstm.ipynb
│   └── bert_fine_tuning_and_attention.ipynb
│
└── generative_and_graphs/
    └── vae_gan_and_gcn_implementations.ipynb
```

---

## Technical Implementations

### 1. Neural Networks from Scratch (NumPy)
- **Path:** `fundamentals/xor_backpropagation_from_scratch.ipynb`
- **Architecture:** Multi-Layer Perceptron (MLP) with 1 hidden layer (3 hidden units) and Sigmoid activation.
- **Concepts:** Manual derivation of the forward pass, mean squared error loss, backpropagation gradient calculations, and stochastic gradient descent optimization.

### 2. Computer Vision (TensorFlow & Keras)
- **Image Classification (CIFAR-10 & MNIST):**
  - **Path:** `computer_vision/cifar10_cnn_classifier.ipynb`, `computer_vision/mnist_digit_classifier.ipynb`
  - **Architecture:** 3-layer Convolutional Neural Network (CNN) with Conv2D, MaxPooling2D, Flatten, and Dense layers. 
- **MRI Brain Tumor Classification:**
  - **Path:** `computer_vision/brain_tumor_mri_detection.ipynb`
  - **Architecture:** Grayscale image preprocessing (224x224x1 resolution), 3-layer CNN architecture with dropout regularizers (0.25 and 0.5) to prevent overfitting, trained via the Adam optimizer with binary cross-entropy loss.

### 3. Natural Language Processing (Keras & PyTorch)
- **LSTM Sequence Classifier:**
  - **Path:** `nlp/twitter_sentiment_lstm.ipynb`
  - **Architecture:** Tokenization and text padding pipeline (10k vocabulary size), Embedding layer (128 dimensions), LSTM layer (128 units), Global Max Pooling, and Dense feedforward layers with dropout. Trained with Early Stopping validation monitoring.
- **BERT Fine-Tuning & Attention Visualization:**
  - **Path:** `nlp/bert_fine_tuning_and_attention.ipynb`
  - **Architecture:** Fine-tuning `bert-base-uncased` (Hugging Face) on text data in PyTorch using the AdamW optimizer. Includes code to extract the attention weights tensor from the final self-attention head and plot word-to-word attention heatmaps.

### 4. Generative & Graph Models (PyTorch)
- **Path:** `generative_and_graphs/vae_gan_and_gcn_implementations.ipynb`
- **Variational Autoencoder (VAE):** Trained on MNIST using a latent space dimension of 2, implementing the reparameterization trick to enable gradient flow during backpropagation.
- **Generative Adversarial Network (GAN):** Mini-max game formulation with a Generator and a Discriminator trained alternately using binary cross-entropy loss.
- **Graph Convolutional Network (GCN):** Semi-supervised node classification using PyTorch Geometric's Graph Convolution layers (`GCNConv`) trained on the Planetoid Cora citation network dataset.
