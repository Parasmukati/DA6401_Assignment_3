import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Layer, Concatenate, Attention
import wandb
from wandb.integration.keras import WandbCallback

# Set GPU memory growth (optional, good practice)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Cannot set memory growth: {e}")

# Data loading
def load_dakshina_data(lang='hi'):
    base_path = f'dakshina_dataset_v1.0/{lang}/lexicons/'
    train_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.train.tsv', sep='\t',
                             header=None, names=['latin', 'native', 'class'])
    val_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.dev.tsv', sep='\t',
                           header=None, names=['latin', 'native', 'class'])
    test_data = pd.read_csv(f'{base_path}{lang}.translit.sampled.test.tsv', sep='\t',
                            header=None, names=['latin', 'native', 'class'])
    return train_data.dropna().astype(str), val_data.dropna().astype(str), test_data.dropna().astype(str)

# Data preprocessing
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
    }

# Bahdanau Attention Layer (Modified for sequence-wise application)
class BahdanauAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query shape: (batch_size, dec_seq_len, hidden_dim)
        # values shape: (batch_size, enc_seq_len, hidden_dim)

        # Expand query to (batch_size, dec_seq_len, 1, hidden_dim)
        query_expanded = tf.expand_dims(query, 2)
        # Apply W2 to query: (batch_size, dec_seq_len, 1, units)
        score_query = self.W2(query_expanded)

        # Expand values to (batch_size, 1, enc_seq_len, hidden_dim)
        values_expanded = tf.expand_dims(values, 1)
        # Apply W1 to values: (batch_size, 1, enc_seq_len, units)
        score_values = self.W1(values_expanded)

        # Sum and apply tanh: (batch_size, dec_seq_len, enc_seq_len, units)
        score = tf.nn.tanh(score_query + score_values)

        # Apply V to score: (batch_size, dec_seq_len, enc_seq_len, 1)
        score = self.V(score)
        # Remove last dimension: (batch_size, dec_seq_len, enc_seq_len)
        score = tf.squeeze(score, axis=-1)

        # Apply softmax along encoder sequence length axis to get attention weights
        # attention_weights shape: (batch_size, dec_seq_len, enc_seq_len)
        attention_weights = tf.nn.softmax(score, axis=-1)

        # Apply attention weights to encoder values
        # context_vector shape: (batch_size, dec_seq_len, enc_seq_len, hidden_dim)
        context_vector = tf.expand_dims(attention_weights, axis=-1) * tf.expand_dims(values, axis=1)

        # Sum along encoder sequence length axis: (batch_size, dec_seq_len, hidden_dim)
        context_vector = tf.reduce_sum(context_vector, axis=2)

        return context_vector, attention_weights

# Attention Seq2Seq Model
class AttentionSeq2Seq:
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 embedding_dim,
                 hidden_dim,
                 dropout_rate=0.2,
                 num_encoder_layers=1, # Keeping simple for fix
                 num_decoder_layers=1): # Keeping simple for fix
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        # _build_model should be called *after* defining self.attention_model
        self.model = self._build_model()


    def _build_model(self):
        encoder_inputs = Input(shape=(None,), name='encoder_input')
        encoder_embedded = Embedding(self.input_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_embedded = Dropout(self.dropout_rate)(encoder_embedded)

        # Encoder LSTM returns sequences & states
        # Assuming single encoder layer for simplicity as in the main block
        # For multi-layer encoder, the last layer should return sequences and states
        # and intermediate layers should return sequences
        encoder_outputs, state_h, state_c = LSTM(self.hidden_dim, return_sequences=True, return_state=True, name='encoder_lstm_0')(encoder_embedded)
        # If multiple encoder layers, collect states from the last layer
        encoder_states = [state_h, state_c]


        decoder_inputs = Input(shape=(None,), name='decoder_input')
        decoder_embedded = Embedding(self.target_vocab_size, self.embedding_dim)(decoder_inputs)
        decoder_embedded = Dropout(self.dropout_rate)(decoder_embedded)

        # Decoder LSTM needs to return sequences to compute attention at each timestep
        # It also needs the initial state from the encoder
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True, name='decoder_lstm_0')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

        # Instantiate the attention layer
        attention_layer = BahdanauAttention(self.hidden_dim)

        # Apply attention: query=decoder_outputs, values=encoder_outputs
        context_vectors, attention_weights = attention_layer(decoder_outputs, encoder_outputs)

        # Concatenate decoder output with context vector
        concat_outputs = Concatenate(axis=-1)([decoder_outputs, context_vectors]) # (batch, dec_seq_len, hidden*2)

        # Final dense layer
        outputs = Dense(self.target_vocab_size, activation='softmax', name='output_dense')(concat_outputs)

        # Define the main prediction model
        self.model = Model([encoder_inputs, decoder_inputs], outputs, name='seq2seq_attention_model')
        # Define a separate model to get the attention weights for plotting
        self.attention_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=attention_weights, name='attention_weights_model')

        return self.model # Return the main prediction model


    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, train_data, val_data, batch_size=64, epochs=10, callbacks=None):
        # Keras Model.fit expects inputs as the first argument (x)
        # and targets as the second argument (y).
        # For models with multiple inputs, x can be a list of arrays or a dictionary
        # mapping input names to arrays. y should be the target output(s).

        # Prepare training inputs as a dictionary matching model input names
        train_inputs = {
            'encoder_input': train_data['encoder_input'],
            'decoder_input': train_data['decoder_input']
        }
        # Prepare training targets
        train_targets = np.expand_dims(train_data['decoder_target'], -1)

        # Prepare validation inputs and targets
        val_inputs = {
            'encoder_input': val_data['encoder_input'],
            'decoder_input': val_data['decoder_input']
        }
        val_targets = np.expand_dims(val_data['decoder_target'], -1)
        validation_data_tuple = (val_inputs, val_targets)

        return self.model.fit(
            x=train_inputs,          # Pass training inputs as x
            y=train_targets,         # Pass training targets as y
            validation_data=validation_data_tuple, # Pass validation data as a tuple
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

# Decoding utility
def decode_sequence(tokenizer, seq):
    reverse_map = {v: k for k, v in tokenizer.word_index.items()}
    reverse_map[0] = ''
    chars = [reverse_map.get(i, '') for i in seq if i != 0]
    return ''.join(chars).replace('\t', '').replace('\n', '').strip()

# Preprocess test data
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

# Evaluate on test set + compute accuracies
def evaluate_model(model, processed_test_data, input_tokenizer, target_tokenizer):
    # Use the main prediction model for evaluation
    preds = model.model.predict([processed_test_data['encoder_input'], processed_test_data['decoder_input']], batch_size=64)
    pred_ids = np.argmax(preds, axis=-1)

    inputs, ground_truths, predictions = [], [], []
    original_target_texts = processed_test_data['original_target_texts']

    for i in range(len(processed_test_data['encoder_input'])):
        inp = decode_sequence(input_tokenizer, processed_test_data['encoder_input'][i])
        gt = decode_sequence(target_tokenizer, target_tokenizer.texts_to_sequences([original_target_texts[i]])[0])
        # Note: The prediction here is from the training model which uses teacher forcing.
        # For true inference prediction, a separate inference model is needed that
        # predicts one token at a time and feeds it back.
        pred = decode_sequence(target_tokenizer, pred_ids[i])
        inputs.append(inp)
        ground_truths.append(gt)
        predictions.append(pred)

    df = pd.DataFrame({'Input': inputs, 'Ground Truth': ground_truths, 'Prediction': predictions})

    # Calculate character and word accuracy
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

    if wandb.run is not None:
        wandb.log({"test_char_accuracy": char_accuracy, "test_word_accuracy": word_accuracy})

    return df

# Log predictions as table and image in wandb
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

# Attention heatmap visualization for sample predictions
def plot_attention_heatmaps(model, processed_test_data, input_tokenizer, target_tokenizer, num_samples=9):
    enc_inputs = processed_test_data['encoder_input']
    dec_inputs = processed_test_data['decoder_input']

    # Use the separate attention_model to get attention weights
    # Predict attention weights for the test set using the attention_model
    attention_weights = model.attention_model.predict([enc_inputs, dec_inputs], batch_size=64)

    for idx in range(min(num_samples, len(enc_inputs))):
        input_seq = enc_inputs[idx]
        dec_input_seq = dec_inputs[idx] # Use decoder input for y-axis length

        # Get the non-padded parts of the sequences for accurate labels
        input_seq_unpadded = [i for i in input_seq if i != 0]
        dec_input_seq_unpadded = [i for i in dec_input_seq if i != 0]

        input_text = decode_sequence(input_tokenizer, input_seq_unpadded)
        dec_input_text = decode_sequence(target_tokenizer, dec_input_seq_unpadded) # Use decoder input for y-axis length

        # Get the attention weights for this specific sample
        # The shape of attn will be (dec_seq_len, enc_seq_len) for this sample
        attn = attention_weights[idx][:len(dec_input_seq_unpadded), :len(input_seq_unpadded)]

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(attn, cmap='viridis')
        fig.colorbar(cax)

        # Set labels corresponding to the characters
        ax.set_xticklabels([''] + list(input_text), rotation=90)
        ax.set_yticklabels([''] + list(dec_input_text))

        ax.set_xlabel('Input Sequence')
        ax.set_ylabel('Output Sequence (Decoder Input)')
        ax.set_title(f'Attention Heatmap for sample {idx + 1}')

        plt.tight_layout()
        plt.show()
        if wandb.run is not None:
             wandb.log({f"attention_heatmap_sample_{idx+1}": wandb.Image(fig)})
        plt.close(fig)


# Main runner
# Main runner
if __name__ == "__main__":
    wandb.init(project="DA_seq2seq_transliteration", name="attention_seq2seq_q5")

    # Load and process data
    train_data, val_data, test_data = load_dakshina_data(lang='hi')
    processed_train_val_data = process_data(train_data, val_data)
    input_tokenizer = processed_train_val_data['input_tokenizer']
    target_tokenizer = processed_train_val_data['target_tokenizer']
    max_in = processed_train_val_data['max_in']
    max_out = processed_train_val_data['max_out']

    processed_test_data = preprocess_test_data(test_data, input_tokenizer, target_tokenizer, max_in, max_out)

    # Use your best hyperparameters or tune again
    best_config = {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "dropout_rate": 0.1,
        "num_encoder_layers": 1,   # For simplicity here just 1 layer
        "num_decoder_layers": 1,
        "input_vocab_size": len(input_tokenizer.word_index) + 1,
        "target_vocab_size": len(target_tokenizer.word_index) + 1,
        "epochs": 10
    }

    model = AttentionSeq2Seq(
        input_vocab_size=best_config['input_vocab_size'],
        target_vocab_size=best_config['target_vocab_size'],
        embedding_dim=best_config['embedding_dim'],
        hidden_dim=best_config['hidden_dim'],
        dropout_rate=best_config['dropout_rate'],
        num_encoder_layers=best_config['num_encoder_layers'],
        num_decoder_layers=best_config['num_decoder_layers']
    )

    model.compile()

    # Train model
    model.fit(
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
        epochs=best_config['epochs'],
        # Explicitly set save_model=False to prevent the deprecated save_format
        callbacks=[WandbCallback(log_model=False, save_graph=False, save_model=False)]
    )

    # Evaluate model on test set
    df_preds = evaluate_model(model, processed_test_data, input_tokenizer, target_tokenizer)

    # Log predictions to wandb
    log_predictions_to_wandb(df_preds)

    # Plot attention heatmaps (you can save or show)
    plot_attention_heatmaps(model, processed_test_data, input_tokenizer, target_tokenizer, num_samples=9)

    wandb.finish()
