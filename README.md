# MiniGPT-J

A minimal generative transformer-based language model implemented entirely from scratch in Java, without the use of any machine learning libraries or automatic differentiation frameworks.

Built as a final year project for B.Sc. (Hons) in Software Development, Atlantic Technological University (ATU), Galway.

[▶ Watch the demo](https://www.youtube.com/watch?v=aZACWW_ihx0)
---

## Project Overview

MiniGPT-J implements the full stack of a character-level language model from first principles:

- Custom matrix engine with forward and backward operations
- Linear layer with manually derived backpropagation
- ReLU activation and cross-entropy loss
- Character-level tokeniser and context-window dataset pipeline
- Token and positional embeddings
- Single-head causal self-attention with full backward pass
- Feed-forward sub-layer with residual connections
- Adam optimiser
- Autoregressive text generation with temperature sampling
- Model serialisation (save/load trained weights)
- Interactive demo script

Trained on Brothers Grimm fairy tales. Achieves a cross-entropy loss of 0.72 over 5000 training steps.

---

## Project Structure

```
app/src/main/java/minigptj/
├── core/
│   ├── Matrix.java           # Core matrix operations
│   ├── Linear.java           # Fully connected layer
│   ├── ReLU.java             # ReLU activation
│   └── LossFunctions.java    # Cross-entropy and MSE loss
├── data/
│   ├── CharTokenizer.java    # Character-level tokeniser
│   ├── TextDataset.java      # Context window dataset
│   ├── OneHot.java           # One-hot encoding (baseline)
│   └── grimm_samples.txt     # Training corpus
├── model/
│   ├── Embedding.java        # Token embedding layer
│   ├── CausalSelfAttention.java  # Single-head causal attention
│   └── MLPLanguageModel.java # MLP baseline model
├── optim/
│   ├── Adam.java             # Adam optimiser
│   └── SGD.java              # SGD optimiser
├── TrainCharLM.java          # Main training script
├── Demo.java                 # Interactive demo
├── ModelIO.java              # Model save/load
└── Runner.java               # Data pipeline sanity check

app/src/test/java/minigptj/  # JUnit 5 unit tests
scripts/PlotResults.py        # Training visualisation script
model.bin                     # Pretrained model checkpoint
```

---

## Prerequisites

- Java 17 or later — [adoptium.net](https://adoptium.net)
- Gradle 9.x — [gradle.org](https://gradle.org)
- Python 3.x with `matplotlib` and `numpy` (visualisations only)

---

## Getting Started

### Clone and Build

```bash
git clone https://github.com/sarahoconnor10/MiniGPT-J.git
cd MiniGPT-J
./gradlew build
```

### Run the Interactive Demo

A pretrained `model.bin` is included. To load it and generate text interactively:

```bash
./gradlew run
```

Type a prompt and press Enter. Type `quit` to exit.

To continue training from the checkpoint for a few hundred steps before generating:

```bash
./gradlew run --args="--train 200"
```

### Run Training from Scratch

```bash
./gradlew run -PmainClass=minigptj.TrainCharLM
```

Training runs for 5000 steps by default and logs loss and a generated sample every 200 steps. On completion, `model.bin` is saved to the project root.

> **Note:** Training takes approximately 60–90 minutes on CPU.

### Run Unit Tests

```bash
./gradlew test
```

All tests should pass. A test report is generated at `app/build/reports/tests/test/index.html`.

### Generate Training Visualisations

```bash
source venv/bin/activate
pip install matplotlib numpy
python3 scripts/PlotResults.py
```

Output PNG files are saved to the project root.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Model dimension (dModel) | 96 |
| Context length | 32 |
| Batch size | 64 |
| Learning rate (Adam) | 0.001 |
| Training steps | 5000 |
| Training corpus | Brothers Grimm fairy tales |
| Vocabulary size | ~65 characters |

---

## Results

| Training Step | Loss |
|---|---|
| 1 | 4.02 |
| 1000 | 1.60 |
| 2000 | 1.16 |
| 3000 | 0.98 |
| 5000 | 0.72 |

Generated text progression (prompt: `"The"`):

- Step 200: `The 'Unarthchhonk. s,' sopen fonwin`
- Step 1000: `The sherwhrecked par sporrrow, 'youn yourte`
- Step 3400: `The shepherd-gone; 'thy cruelty shall so`
- Step 5000: `The shep-grane.' All the wine s all cost take all up.`

---

## Author

Sarah O'Connor  
B.Sc. (Hons) in Software Development  
Atlantic Technological University (ATU), Galway  
Supervisor: Gerard Harrison