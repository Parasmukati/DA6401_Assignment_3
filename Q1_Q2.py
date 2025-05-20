import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import wandb
from wandb.keras import WandbCallback

# Authenticate with wandb
wandb.login(key='b07291a5fa9f067521980dccc1f417e554f9fd4b')

def read_dataset(file_path):
    data_frame = pd.read_csv(file_path, sep='\t', header=None, names=['latin', 'native'])
    data_frame = data_frame.dropna()
    data_frame['latin'] = data_frame['latin'].astype(str)
    data_frame['native'] = data_frame['native'].astype(str)
    return data_frame

def get_dakshina_data(lang='hi', base_folder='dakshina_dataset_v1.0'):
    lexicons_dir = os.path.join(base_folder, lang, 'lexicons')
    train_fp = os.path.join(lexicons_dir, f'{lang}.translit.sampled.train.tsv')
    val_fp = os.path.join(lexicons_dir, f'{lang}.translit.sampled.dev.tsv')
    test_fp = os.path.join(lexicons_dir, f'{lang}.translit.sampled.test.tsv')
    return read_dataset(train_fp), read_dataset(val_fp), read_dataset(test_fp)

# Load the data splits
train_df, val_df, test_df = get_dakshina_data()

# Prepare text sequences with start/end tokens
train_input_texts = train_df['latin'].tolist()
train_target_texts = ['\t' + text + '\n' for text in train_df['native']]

val_input_texts = val_df['latin'].tolist()
val_target_texts = ['\t' + text + '\n' for text in val_df['native']]

# Character level tokenization
input_tokenizer = Tokenizer(char_level=True)
input_tokenizer.fit_on_texts(train_input_texts + val_input_texts)

target_tokenizer = Tokenizer(char_level=True)
target_tokenizer.fit_on_texts(train_target_texts + val_target_texts)

# Calculate max sequence lengths for padding
max_encoder_seq_len = max(len(seq) for seq in train_input_texts + val_input_texts)
max_decoder_seq_len = max(len(seq) for seq in train_target_texts + val_target_texts)

def encode_and_pad(tokenizer, texts, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

# Encode & pad encoder input
encoder_input_train = encode_and_pad(input_tokenizer, train_input_texts, max_encoder_seq_len)
encoder_input_val = encode_and_pad(input_tokenizer, val_input_texts, max_encoder_seq_len)

# Encode & pad decoder input & target sequences
decoder_seq_train = encode_and_pad(target_tokenizer, train_target_texts, max_decoder_seq_len)
decoder_seq_val = encode_and_pad(target_tokenizer, val_target_texts, max_decoder_seq_len)

# For training decoder targets, shift decoder inputs by one timestep
decoder_input_train = decoder_seq_train[:, :-1]
decoder_target_train = decoder_seq_train[:, 1:]

decoder_input_val = decoder_seq_val[:, :-1]
decoder_target_val = decoder_seq_val[:, 1:]

# Setup wandb experiment configuration
try:
    wandb.finish()
except Exception:
    pass

try:
    wandb.init(
        project="DA_seq2seq_transliteration",
        name="vanilla_lstm_run_q1_refactored",
        config={
            "model": "vanilla_seq2seq",
            "rnn_cell": "LSTM",
            "embedding_dim": 64,
            "hidden_units": 128,
            "dropout": 0.2,
            "batch_size": 64,
            "epochs": 10,
            "input_vocab_size": len(input_tokenizer.word_index) + 1,
            "target_vocab_size": len(target_tokenizer.word_index) + 1,
            "max_encoder_length": max_encoder_seq_len,
            "max_decoder_length": max_decoder_seq_len,
            "optimizer": "adam",
            "loss_function": "sparse_categorical_crossentropy"
        }
    )
except Exception as e:
    print(f"Wandb init failed: {e}")
    class DummyWandb:
        def log(self, *args, **kwargs): pass
        def finish(self): pass
        @property
        def config(self): 
            return {
                'embedding_dim': 64,
                'hidden_units': 128,
                'batch_size': 64,
                'epochs': 10,
                'num_encoder_layers': 1,
                'num_decoder_layers': 1
            }
    wandb = DummyWandb()

config = wandb.config
D = config['embedding_dim']
H = config['hidden_units']
batch_size = config.get('batch_size', 64)
epochs = config.get('epochs', 10)
num_enc_layers = config.get('num_encoder_layers', 1)
num_dec_layers = config.get('num_decoder_layers', 1)

# Define the Seq2Seq Model class
class Seq2SeqModel:
    def __init__(self,
                 encoder_vocab_size,
                 decoder_vocab_size,
                 embedding_dim,
                 hidden_units,
                 rnn_type='LSTM',
                 dropout_rate=0.2,
                 encoder_layers=1,
                 decoder_layers=1):
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.rnn_type = rnn_type
        self.dropout_rate = dropout_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.model = self._construct_model()

    def _rnn_layer(self, units, return_sequences=True, return_state=True):
        if self.rnn_type == 'LSTM':
            return LSTM(units, return_sequences=return_sequences, return_state=return_state)
        elif self.rnn_type == 'GRU':
            return GRU(units, return_sequences=return_sequences, return_state=return_state)
        else:
            return tf.keras.layers.SimpleRNN(units, return_sequences=return_sequences, return_state=return_state)

    def _construct_model(self):
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        x = Embedding(self.encoder_vocab_size, self.embedding_dim, name="encoder_embedding")(encoder_inputs)
        x = Dropout(self.dropout_rate)(x)

        encoder_states = None
        # Stack multiple encoder RNN layers
        for i in range(self.encoder_layers):
            # For all but last layer, return sequences=True to feed next layer
            return_seq = True if i < self.encoder_layers - 1 else False
            if self.rnn_type == 'LSTM':
                x, state_h, state_c = LSTM(
                    self.hidden_units,
                    return_sequences=return_seq,
                    return_state=True,
                    name=f"encoder_lstm_{i}"
                )(x)
                encoder_states = [state_h, state_c]
            else:
                x, state_h = self._rnn_layer(
                    self.hidden_units,
                    return_sequences=return_seq,
                    return_state=True
                )(x)
                encoder_states = [state_h]

        decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        y = Embedding(self.decoder_vocab_size, self.embedding_dim, name="decoder_embedding")(decoder_inputs)
        y = Dropout(self.dropout_rate)(y)

        # Stack decoder layers; initial state only fed to first decoder layer
        for i in range(self.decoder_layers):
            return_seq = True  # decoder outputs sequence always
            if self.rnn_type == 'LSTM':
                initial_state = encoder_states if i == 0 else None
                if initial_state is not None:
                    y, dec_state_h, dec_state_c = LSTM(
                        self.hidden_units,
                        return_sequences=return_seq,
                        return_state=True,
                        name=f"decoder_lstm_{i}"
                    )(y, initial_state=initial_state)
                else:
                    y, dec_state_h, dec_state_c = LSTM(
                        self.hidden_units,
                        return_sequences=return_seq,
                        return_state=True,
                        name=f"decoder_lstm_{i}"
                    )(y)
            else:
                initial_state = encoder_states if i == 0 else None
                if initial_state is not None:
                    y, dec_state_h = self._rnn_layer(
                        self.hidden_units,
                        return_sequences=return_seq,
                        return_state=True
                    )(y, initial_state=initial_state)
                else:
                    y, dec_state_h = self._rnn_layer(
                        self.hidden_units,
                        return_sequences=return_seq,
                        return_state=True
                    )(y)

        output_probs = Dense(self.decoder_vocab_size, activation='softmax', name="output_projection")(y)
        return Model([encoder_inputs, decoder_inputs], output_probs)

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, train_dataset, val_dataset, batch_size=64, epochs=10, callbacks=None):
        return self.model.fit(
            [train_dataset['encoder_input'], train_dataset['decoder_input']],
            np.expand_dims(train_dataset['decoder_target'], axis=-1),
            validation_data=(
                [val_dataset['encoder_input'], val_dataset['decoder_input']],
                np.expand_dims(val_dataset['decoder_target'], axis=-1)
            ),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

# Instantiate and compile model
seq2seq = Seq2SeqModel(
    encoder_vocab_size=len(input_tokenizer.word_index) + 1,
    decoder_vocab_size=len(target_tokenizer.word_index) + 1,
    embedding_dim=D,
    hidden_units=H,
    rnn_type='LSTM',
    dropout_rate=0.2,
    encoder_layers=num_enc_layers,
    decoder_layers=num_dec_layers
)
seq2seq.compile_model()

# Calculate FLOPs estimate (same formula but rewritten)
steps_encoder = encoder_input_train.shape[1]
steps_decoder = decoder_input_train.shape[1]
flops_per_timestep = 4 * (H * D + H * H)
flops_encoder = steps_encoder * num_enc_layers * flops_per_timestep
flops_decoder = steps_decoder * num_dec_layers * flops_per_timestep
total_flops_estimate = flops_encoder + flops_decoder

print(f"Estimated total multiply-accumulate operations (encoder + decoder): {total_flops_estimate:,}")
print(f"Total trainable parameters: {seq2seq.model.count_params():,}")
seq2seq.model.summary()

# Setup wandb callback safely
try:
    wandb_cb = WandbCallback(log_model=False, save_graph=False, save_model=False)
    callbacks_list = [wandb_cb]
except Exception as err:
    print(f"WandbCallback init failed: {err}")
    callbacks_list = []

# Train model
history = seq2seq.train(
    train_dataset={
        'encoder_input': encoder_input_train,
        'decoder_input': decoder_input_train,
        'decoder_target': decoder_target_train
    },
    val_dataset={
        'encoder_input': encoder_input_val,
        'decoder_input': decoder_input_val,
        'decoder_target': decoder_target_val
    },
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks_list
)

# Log metrics to wandb with safety
try:
    wandb.log({
        "estimated_flops": total_flops_estimate,
        "trainable_parameters": seq2seq.model.count_params()
    })
    wandb.finish()
except Exception as err:
    print(f"Error logging to wandb: {err}")
