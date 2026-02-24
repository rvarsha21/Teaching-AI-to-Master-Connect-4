# Teaching-AI-to-Master-Connect-4
Learning Strategic Play with CNNs and Transformers

## Project Overview

This project trains deep learning models to predict optimal moves in **Connect 4** by imitating high-quality Monte Carlo Tree Search (MCTS) recommendations.

Instead of using brute-force search during gameplay, we train:

* Convolutional Neural Networks (CNNs)
* Transformer-based architectures

The goal is to understand how different neural architectures learn structured board-state strategy and how data quality impacts performance.

---

## Objectives

* Predict optimal next move (7-class classification: columns 0–6)
* Imitate high-quality MCTS (up to 10,000 rollouts per move)
* Compare CNN vs Transformer architectures
* Prevent data leakage using game-level splitting
* Combine neural predictions with tactical rule-based logic

---

## Project Structure

```
├── connect4_engine.py          # Game logic
├── mcts.py                     # Monte Carlo Tree Search
├── generate_data.py            # Self-play data generation
├── preprocessing.ipynb         # Canonical encoding + splits
├── cnn_models.ipynb            # CNN architectures
├── transformer_models.ipynb    # Transformer architectures
├── play_vs_ai.py               # Human vs AI interface
└── logs/                       # Training metrics
```

---

## Dataset

* Generated via self-play using MCTS
* Final high-quality dataset: ~32,000 games
* MCTS rollouts per move: **10,000**
* Board encoding: `(6 × 7 × 2)`

  * Channel 0 → Current player
  * Channel 1 → Opponent
* Stratified split:

  * 80% Train
  * 10% Validation
  * 10% Test

### Key Insight

Data quality mattered more than data quantity.
60,000 games with weak MCTS (800 steps) performed worse than fewer games with 10,000-step MCTS.

---

# Convolutional Neural Networks

We experimented with progressively deeper CNN architectures.

### Best Model — Tactical CNN

* ~2M parameters
* Game-level split (no leakage)
* Canonical perspective encoding
* Tactical inference wrapper (handles immediate wins/blocks)

**Performance**

* Best Validation Accuracy: **72.07%**
* Stable training with minimal overfitting

### Why It Worked

* High-quality MCTS labels
* Clean splitting strategy
* Hybrid inference (rules + CNN)

CNNs performed best for structured spatial board-state learning.

---

# Transformers

We tested whether attention mechanisms alone could learn board strategy.

### Model 1 — Basic Transformer

* No positional encoding
* Test Accuracy: **21.96%**
* Failed due to lack of spatial awareness

### Model 2 — Positional Transformer

* Added learned positional embeddings
* Test Accuracy: **55.12%**

### Model 3 — Deep Transformer (6 blocks)

* ~500K parameters
* Test Accuracy: **60.87%**

### Key Insight

Positional information was critical.
Adding positional embeddings improved accuracy by +33%.

However, Transformers underperformed CNNs on this structured grid task.

---

## CNN vs Transformer Comparison

| Model            | Test Accuracy |
| ---------------- | ------------- |
| Baseline CNN     | 68%           |
| ResNet CNN       | 69%           |
| Tactical CNN     | **72%**       |
| Deep Transformer | 61%           |

CNNs were better suited for localized spatial pattern detection in Connect 4.

---

## Major Lessons Learned

* **Data quality > data quantity**
* **Game-level splitting prevents leakage**
* Larger models do not guarantee better generalization
* Positional encoding is essential for Transformers
* Hybrid systems (NN + rule-based logic) outperform pure models

---

## Future Improvements

* Reinforcement learning beyond imitation
* CNN + Transformer ensemble
* Neural-guided lightweight MCTS
* Curriculum learning from simple to complex board states

---

## Tech Stack

* Python
* NumPy
* TensorFlow / Keras
* PyTorch
* Numba (JIT optimization)
* Multiprocessing

---

## Final Takeaway

Building a strong Connect 4 AI is harder than it looks.

Even with:

* High-quality MCTS supervision
* Deep architectures
* Millions of board states

There remains a large gap between predicting the correct move ~70% of the time and consistently winning games.

Strategic reasoning requires both learned pattern recognition and tactical precision.

* Shorten further for portfolio highlight
* Or convert into a resume project bullet section 🚀

## Live Demo

The application is deployed on AWS EC2.

⚠️ Note: The demo may be temporarily unavailable if the instance is stopped to optimize cloud costs.

Live URL: https://msba25optim2-24.anvil.app/
