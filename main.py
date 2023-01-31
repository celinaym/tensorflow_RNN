import tensorflow as tf
import numpy as np
import os
import time

window_size = 100
shuffle_buffer = 10000
batch_size = 64
embedding_dim = 256
rnn_units = 1024

# TODO download Shakespeare datasets
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# TODO print unique characters
vocab = sorted(set(text))
vocab_size = len(vocab)

# TODO Preprocessing
# TODO STEP1) Make Character Dictionary
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# TODO STEP2) convert text into int
text_as_int = np.array([char2idx[c] for c in text])


def windowed_dataset(series, window_size, shuffle_buffer, batch_size):
    series = tf.expand_dims(series, -1)
    ds = tf.data.Dataset.from_tensor_slices(series)     # convert into Dataset
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x.batch(window_size+1))
    ds = ds.shuffle(shuffle_buffer)
    # TODO map trained data/target data(could be many-many, many-one etc)
    ds = ds.map(lambda x: (x[:-1], x[1:]))
    return ds.batch(batch_size).prefetch(1)


# TODO STEP3) create X,Y datasets(will use Windowed Dataset)
train_data = windowed_dataset(np.array(text_as_int), window_size, shuffle_buffer, batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,  # True for many-to-many, False for many-to-one
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
model.summary()

# TODO STEP4) create checkpoint
checkpoint_path = './models/my_checkpt.ckpt'

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss',
    verbose=1,
)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss, metrics=['acc'])
model.fit(train_data,
          epochs=10,
          steps_per_epoch=1720,
          callbacks=[checkpoint_callback])

# TODO STEP5) Redefine model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[1, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])
model.load_weights(checkpoint_path)

model.build(tf.TensorShape([1, None]))
model.summary()


# TODO STEP6) Evaluation Stage
def generate_text(model, start_string):

    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 결과를 저장할 빈 문자열
    text_generated = []

    # TODO As temperature decreases, more predictable
    temperature = 1.0

    # TODO batch_size = 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # TODO remove dimension(batch_size)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # TODO predicted output as a new input(with a previous Hidden layer state)
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


print(generate_text(model, start_string=u"ROMEO: "))