import os
import random
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

DATASET_PATH = 'dataset'

# Helper to load dataset

def load_dataset(path=DATASET_PATH):
    texts = []
    labels = []
    poets = os.listdir(path)
    for poet in poets:
        poet_dir = os.path.join(path, poet)
        if not os.path.isdir(poet_dir):
            continue
        for poem in os.listdir(poet_dir):
            poem_dir = os.path.join(poet_dir, poem)
            class_file = os.path.join(poem_dir, 'CLASS.txt')
            text_file = os.path.join(poem_dir, f'{poem}.txt')
            if os.path.isfile(class_file) and os.path.isfile(text_file):
                with open(class_file, encoding='utf-8') as f:
                    label = f.read().strip()
                with open(text_file, encoding='utf-8') as f:
                    text = f.read().strip()
                texts.append(text)
                labels.append(label)
    return texts, labels


def prepare_datasets(texts, labels, test_size=0.2, batch_size=32, max_tokens=20000, seq_len=200):
    label_set = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    y = [label_to_index[label] for label in labels]
    x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=test_size, random_state=42, stratify=y)

    vectorizer = layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_len)
    text_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
    vectorizer.adapt(text_ds)

    def vectorize(text, label):
        text = tf.expand_dims(text, -1)
        return vectorizer(text), label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(1024).batch(batch_size).map(vectorize).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).map(vectorize).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, len(label_set), vectorizer


def build_model(num_classes, vocab_size=20000, seq_len=200, embedding_dim=128):
    inputs = layers.Input(shape=(seq_len,))
    x = layers.Embedding(vocab_size, embedding_dim)(inputs)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    texts, labels = load_dataset()
    train_ds, test_ds, num_classes, vectorizer = prepare_datasets(texts, labels)
    model = build_model(num_classes)
    epochs = int(os.environ.get('EPOCHS', '5'))
    model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    model.save('poem_classifier.h5')
