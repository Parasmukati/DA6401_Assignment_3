import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Dropout
import os
import time

# --- Ensure TensorFlow GPU memory growth ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Cannot set memory growth: {e}")

# --- Data Loading ---
def load_dakshina_data(lang='hi'):
    base_path = f'dakshina_dataset_v1.0/{lang}/lexicons/'
    train_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.train.tsv', sep='\t',
                             header=None, names=['latin', 'native', 'class'])
    val_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.dev.tsv', sep='\t',
                           header=None, names=['latin', 'native', 'class'])
    test_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.test.tsv', sep='\t',
                            header=None, names=['latin', 'native', 'class'])
    # Drop missing values
    return train_data.dropna().astype(str), val_data.dropna().astype(str), test_data.dropna().astype(str)

def process_data(train_data, val_data):
    input_texts = train_data['latin'].tolist()
    target_texts = ['\t' + t + '\n' for t in train_data['native'].tolist()]

    val_input_texts = val_data['latin'].tolist()
    val_target_texts = ['\t' + t + '\n' for t in val_data['native'].tolist()]

    input_tokenizer = Tokenizer(char_level=True, oov_token=None)
    input_tokenizer.fit_on_texts(input_texts + val_input_texts)

    target_tokenizer = Tokenizer(char_level=True, oov_token=None)
    target_tokenizer.fit_on_texts(target_texts + val_target_texts)

    max_in = max(len(txt) for txt in input_texts + val_input_texts)
    max_out = max(len(txt) for txt in target_texts + val_target_texts)

    encoder_input_train = pad_sequences(input_tokenizer.texts_to_sequences(input_texts), maxlen=max_in, padding='post')
    decoder_input_train = pad_sequences(target_tokenizer.texts_to_sequences(target_texts), maxlen=max_out, padding='post')
    decoder_target_train = np.array(decoder_input_train)[:, 1:]
    decoder_input_train = np.array(decoder_input_train)[:, :-1]

    encoder_input_val = pad_sequences(input_tokenizer.texts_to_sequences(val_input_texts), maxlen=max_in, padding='post')
    decoder_input_val = pad_sequences(target_tokenizer.texts_to_sequences(val_target_texts), maxlen=max_out, padding='post')
    decoder_target_val = np.array(decoder_input_val)[:, 1:]
    decoder_input_val = np.array(decoder_input_val)[:, :-1]

    return {
        'input_tokenizer': input_tokenizer,
        'target_tokenizer': target_tokenizer,
        'max_in': max_in,
        'max_out': max_out,
        'encoder_input_train': encoder_input_train,
        'decoder_input_train': decoder_input_train,
        'decoder_target_train': decoder_target_train,
        'encoder_input_val': encoder_input_val,
        'decoder_input_val': decoder_input_val,
        'decoder_target_val': decoder_target_val,
        'input_texts': input_texts,
        'target_texts': target_texts,
        'val_input_texts': val_input_texts,
        'val_target_texts': val_target_texts
    }

# --- Model Definition ---
class VanillaSeq2Seq:
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 embedding_dim,
                 hidden_dim,
                 cell_type='LSTM',
                 dropout_rate=0.2,
                 num_encoder_layers=1,
                 num_decoder_layers=1):
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.dropout_rate = dropout_rate
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model = self._build_model()

    def _rnn_layer(self, return_sequences, return_state):
        if self.cell_type == 'LSTM':
            return LSTM(self.hidden_dim, return_sequences=return_sequences, return_state=return_state)
        elif self.cell_type == 'GRU':
            return GRU(self.hidden_dim, return_sequences=return_sequences, return_state=return_state)
        else:
            return SimpleRNN(self.hidden_dim, return_sequences=return_sequences, return_state=return_state)

    def _build_model(self):
        encoder_inputs = Input(shape=(None,), name='encoder_input')
        x = Embedding(self.input_vocab_size, self.embedding_dim)(encoder_inputs)
        x = Dropout(self.dropout_rate)(x)

        encoder_states = []
        for i in range(self.num_encoder_layers):
            rs = (i < self.num_encoder_layers - 1)
            if self.cell_type == 'LSTM':
                if i == self.num_encoder_layers - 1:
                    x, state_h, state_c = LSTM(self.hidden_dim, return_sequences=rs, return_state=True, name=f'enc_lstm_{i}')(x)
                    encoder_states = [state_h, state_c]
                else:
                    x, _, _ = LSTM(self.hidden_dim, return_sequences=rs, return_state=True, name=f'enc_lstm_{i}')(x)
            else:
                if i == self.num_encoder_layers - 1:
                    x, state_h = self._rnn_layer(return_sequences=rs, return_state=True)(x)
                    encoder_states = [state_h]
                else:
                    x, _ = self._rnn_layer(return_sequences=rs, return_state=True)(x)

        decoder_inputs = Input(shape=(None,), name='decoder_input')
        y = Embedding(self.target_vocab_size, self.embedding_dim)(decoder_inputs)
        y = Dropout(self.dropout_rate)(y)

        decoder_outputs_from_prev_layer = y
        decoder_states_outputs = []

        for i in range(self.num_decoder_layers):
            return_sequences = True
            return_state = True
            layer_input = decoder_outputs_from_prev_layer

            if self.cell_type == 'LSTM':
                initial_state = encoder_states if i == 0 else None
                lstm_layer = LSTM(self.hidden_dim, return_sequences=return_sequences, return_state=return_state, name=f'dec_lstm_{i}')
                if initial_state:
                    decoder_outputs_from_prev_layer, state_h, state_c = lstm_layer(layer_input, initial_state=initial_state)
                else:
                    decoder_outputs_from_prev_layer, state_h, state_c = lstm_layer(layer_input)
                decoder_states_outputs.append([state_h, state_c])
            else:
                initial_state = encoder_states[0] if i == 0 and encoder_states else None
                rnn_layer = self._rnn_layer(return_sequences=return_sequences, return_state=return_state)
                if initial_state is not None:
                    decoder_outputs_from_prev_layer, state_h = rnn_layer(layer_input, initial_state=[initial_state])
                else:
                    decoder_outputs_from_prev_layer, state_h = rnn_layer(layer_input)
                decoder_states_outputs.append([state_h])

        outputs = Dense(self.target_vocab_size, activation='softmax')(decoder_outputs_from_prev_layer)

        return Model([encoder_inputs, decoder_inputs], outputs)

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, train_data, val_data, batch_size=64, epochs=10, callbacks=None):
        return self.model.fit(
            [train_data['encoder_input'], train_data['decoder_input']],
            np.expand_dims(train_data['decoder_target'], -1),
            validation_data=(
                [val_data['encoder_input'], val_data['decoder_input']],
                np.expand_dims(val_data['decoder_target'], -1)
            ),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

# --- Decoding & Evaluation ---
def decode_sequence(tokenizer, seq):
    reverse_map = {v: k for k, v in tokenizer.word_index.items()}
    reverse_map[0] = ''
    chars = [reverse_map.get(i, '') for i in seq if i != 0]
    return ''.join(chars).replace('\t', '').replace('\n', '').strip()

def preprocess_test_data(test_df, input_tokenizer, target_tokenizer, max_in, max_out):
    input_texts = test_df['latin'].tolist()
    target_texts = ['\t' + t + '\n' for t in test_df['native'].tolist()]

    encoder_input_test = pad_sequences(input_tokenizer.texts_to_sequences(input_texts), maxlen=max_in, padding='post')
    decoder_input_test = pad_sequences(target_tokenizer.texts_to_sequences(target_texts), maxlen=max_out, padding='post')
    decoder_target_test = np.array(decoder_input_test)[:, 1:]
    decoder_input_test = np.array(decoder_input_test)[:, :-1]

    return {
        'encoder_input': encoder_input_test,
        'decoder_input': decoder_input_test,
        'decoder_target': decoder_target_test,
        'original_target_texts': target_texts
    }

def evaluate_model(model, processed_test_data, input_tokenizer, target_tokenizer):
    preds = model.model.predict([processed_test_data['encoder_input'], processed_test_data['decoder_input']], batch_size=64)
    pred_ids = np.argmax(preds, axis=-1)

    inputs = []
    ground_truths = []
    predictions = []
    original_target_texts = processed_test_data['original_target_texts']

    for i in range(len(processed_test_data['encoder_input'])):
        inp = decode_sequence(input_tokenizer, processed_test_data['encoder_input'][i])
        gt = decode_sequence(target_tokenizer, target_tokenizer.texts_to_sequences([original_target_texts[i]])[0])
        pred = decode_sequence(target_tokenizer, pred_ids[i])

        inputs.append(inp)
        ground_truths.append(gt)
        predictions.append(pred)

    df = pd.DataFrame({
        'Input': inputs,
        'Ground Truth': ground_truths,
        'Prediction': predictions
    })

    # Character-level and word-level accuracy
    char_correct = 0
    total_chars = 0
    word_correct = 0
    total_words = len(df)

    for _, row in df.iterrows():
        gt = row['Ground Truth']
        pred = row['Prediction']
        min_len = min(len(gt), len(pred))
        char_correct += sum(gt[j] == pred[j] for j in range(min_len))
        total_chars += max(len(gt), len(pred))
        if gt == pred:
            word_correct += 1

    char_accuracy = char_correct / total_chars if total_chars > 0 else 0
    word_accuracy = word_correct / total_words if total_words > 0 else 0

    print(f"Test Character Accuracy: {char_accuracy:.4f}")
    print(f"Test Word Accuracy (Exact Match): {word_accuracy:.4f}")

    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "test_char_accuracy": char_accuracy,
            "test_word_accuracy": word_accuracy
        })

    return df

def log_predictions_to_wandb(df, num_samples=10):
    table = wandb.Table(dataframe=df.head(num_samples))
    wandb.log({"predictions_table": table})

    fig, ax = plt.subplots(figsize=(12, num_samples * 0.6))
    ax.axis('off')
    table_data = df.head(num_samples)[['Input', 'Ground Truth', 'Prediction']].values
    table_data = np.vstack([['Input', 'Ground Truth', 'Prediction'], table_data])

    table_plot = ax.table(cellText=table_data, cellLoc='left', loc='center', colWidths=[0.3,0.3,0.4])
    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(10)
    table_plot.scale(1.2, 1.2)

    for (row, col), cell in table_plot._cells.items():
        if row == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
            cell.set_text_props(ha='center', va='center', weight='bold')

    plt.tight_layout()
    wandb.log({"predictions_image": wandb.Image(fig)})
    plt.close(fig)

if __name__ == "__main__":
    # Initialize wandb for evaluation
    if wandb.run is None:
        wandb.init(project="DA_seq2seq_transliteration", name="Q4_evaluation")

    # Load data and process
    train_data, val_data, test_df = load_dakshina_data(lang='hi')
    processed_train_val_data = process_data(train_data, val_data)
    input_tokenizer = processed_train_val_data['input_tokenizer']
    target_tokenizer = processed_train_val_data['target_tokenizer']
    max_in = processed_train_val_data['max_in']
    max_out = processed_train_val_data['max_out']

    processed_test_data = preprocess_test_data(test_df, input_tokenizer, target_tokenizer, max_in, max_out)

    # Best hyperparameters from sweep
    best_config = {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "cell_type": "LSTM",
        "dropout_rate": 0.1,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "input_vocab_size": len(input_tokenizer.word_index) + 1,
        "target_vocab_size": len(target_tokenizer.word_index) + 1,
        "epochs": 10
    }

    # Build and compile model
    model = VanillaSeq2Seq(
        input_vocab_size=best_config['input_vocab_size'],
        target_vocab_size=best_config['target_vocab_size'],
        embedding_dim=best_config['embedding_dim'],
        hidden_dim=best_config['hidden_dim'],
        cell_type=best_config['cell_type'],
        dropout_rate=best_config['dropout_rate'],
        num_encoder_layers=best_config['num_encoder_layers'],
        num_decoder_layers=best_config['num_decoder_layers']
    )
    model.compile()

    # Train model from scratch on train+val data (optional: concatenate datasets for more training)
    # Here using train and val separately for validation monitoring
    history = model.fit(
        train_data={
            'encoder_input': processed_train_val_data['encoder_input_train'],
            'decoder_input': processed_train_val_data['decoder_input_train'],
            'decoder_target': processed_train_val_data['decoder_target_train']
        },
        val_data={
            'encoder_input': processed_train_val_data['encoder_input_val'],
            'decoder_input': processed_train_val_data['decoder_input_val'],
            'decoder_target': processed_train_val_data['decoder_target_val']
        },
        batch_size=64,
        epochs=best_config['epochs']
    )

    # Evaluate on test set
    df_predictions = evaluate_model(model, processed_test_data, input_tokenizer, target_tokenizer)

    # Log predictions to wandb
    log_predictions_to_wandb(df_predictions)

    print("Q4 Evaluation complete. Check WandB for results.")
    wandb.finish()
