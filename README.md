# DA6401 Assignment 3 - Seq2Seq Transliteration Model
NA21B051
Paras Mukati

---
## Project Overview

The goal is to train a vanilla Seq2Seq model that converts Latin script text sequences into Hindi native script. Key features include:

- Character-level tokenization of both input and output sequences.
- Embedding layers to learn dense vector representations.
- Configurable encoder and decoder with LSTM or GRU cells.
- Dropout regularization for robustness.
- Teacher forcing during training with shifted target sequences.
- FLOPs and parameter count estimations.
- Integration with Weights & Biases for experiment logging.

---

## Dataset

The project uses the **Dakshina Dataset v1.0** by Google Research:

- Language: Hindi (`hi`)
- Data splits: Train, Validation (dev), and Test sets
- Format: TSV files with two columns — Latin script text and corresponding native script transliteration

Download the dataset from [Google Research Dakshina GitHub](https://github.com/google-research-dakshina).

Place the dataset in the following directory structure:

```
dakshina_dataset_v1.0/
 └── hi/
     └── lexicons/
         ├── hi.translit.sampled.train.tsv
         ├── hi.translit.sampled.dev.tsv
         └── hi.translit.sampled.test.tsv
```

---

## Installation

### Prerequisites

- Python 3.7 or above
- TensorFlow 2.x
- pandas
- numpy
- wandb

### Install dependencies

```bash
pip install tensorflow pandas numpy wandb
```

---

## Usage

1. **Login to Weights & Biases**

Before training, authenticate with wandb:

```python
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")
```

2. **Run the training script**

```bash
python train.py
```

The script will:

- Load and preprocess the dataset.
- Tokenize and pad sequences at the character level.
- Build and compile the Seq2Seq model.
- Train the model using teacher forcing.
- Track metrics and hyperparameters using WandB.

---

## Model Architecture

- **Encoder**: Embedding layer followed by configurable stacked LSTM or GRU layers that produce context states.
- **Decoder**: Embedding layer and stacked recurrent layers that receive the encoder’s final states as initial states.
- **Output Layer**: Dense layer with softmax activation predicting probabilities over the target vocabulary characters.

---

## Configuration

Model and training hyperparameters can be configured via the script or wandb config:

| Parameter            | Description                           | Default |
|----------------------|-------------------------------------|---------|
| `rnn_cell`           | Recurrent cell type (`LSTM`, `GRU`) | `LSTM`  |
| `embedding_dim`      | Dimension of embeddings              | 64      |
| `hidden_units`       | Number of units per RNN layer        | 128     |
| `dropout`            | Dropout rate                        | 0.2     |
| `batch_size`         | Number of samples per batch          | 64      |
| `epochs`             | Number of training epochs            | 10      |
| `num_encoder_layers` | Number of encoder RNN layers         | 1       |
| `num_decoder_layers` | Number of decoder RNN layers         | 1       |

---

## Training and Evaluation

- **Loss:** Sparse Categorical Crossentropy  
- **Optimizer:** Adam  
- **Teacher Forcing:** Decoder input sequences are the target sequences shifted by one timestep.  
- **Validation:** Monitored during training for performance and overfitting.

---

## Performance Metrics

- Estimated floating point operations (FLOPs) are computed based on model parameters and sequence lengths.
- Total trainable parameters are printed to indicate model complexity.
- Training and validation accuracy and loss metrics are logged and monitored.

---

## Experiment Tracking

Integrated with [Weights & Biases](https://wandb.ai/) for:

- Automatic logging of hyperparameters and configuration
- Real-time tracking of training and validation metrics
- Visualization of model performance

Set up a free WandB account and login before starting training.

---

## Folder Structure

```
.
├── Q1_Q2.py         
├── Q4.py  
├── Q5.py             
├── dakshina_dataset_v1.0/      # Dataset folder with language files
├── requirements.txt            # Python dependencies
├── README.md                   # This README file
```

---

## References

- [Dakshina Dataset](https://github.com/google-research-dakshina)  
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. https://arxiv.org/abs/1409.3215  
- [Weights & Biases Documentation](https://wandb.ai/na21b051-indian-institute-of-technology-madras/DA_seq2seq_transliteration/reports/DA6401-Assignment-3--VmlldzoxMjg2Mzk2MQ?accessToken=n2jwxobfm9ubuk83jmbsgl4g0ye7qzc1anfux61kiceb4b0l3y7d1uqfvwuc9ip4)

---
