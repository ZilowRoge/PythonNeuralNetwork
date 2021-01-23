import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image, ImageOps

TRAIN_DIR = 'Train/'
TEST_DIR = 'Test/'

# learning_rate = 0.001
# training_epoch = 500
# batch_size = 100
# display_step = 1
#
# #Network Parameters
# n_input = 1024
# n_hidden_1 = 512
# n_hidden_2 = 64
# n_classes = 8


def load_dataset(directory):
    dataset = ([], [])
    number_of_classes = len(os.listdir(directory))

    for filename in os.listdir(directory):
        for img_name in os.listdir(os.path.join(directory, filename)):
            img = cv2.imread(os.path.join(directory, filename, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                dataset[0].append(np.array(img))
                dataset[1].append(int(filename))

    return dataset


def get_batch(x_data, y_data, batch_size):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs, :, :], y_data[idxs]


def nn_model(x_input, W1, b1, W2, b2):
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, W2), b2)
    return logits


def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy

def train():
    epochs = 10
    batch_size = 200
    optimalizer = tf.keras.optimizers.Adam()

    (x_train, y_train) = load_dataset(TRAIN_DIR)
    (x_test, y_test) = load_dataset(TEST_DIR)

    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    x_test = tf.Variable(x_test)

    W1 = tf.Variable(tf.random.normal([1024, 400], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([400]), name='b1')

    W2 = tf.Variable(tf.random.normal([400, 9], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([9]), name='b1')

    total_batch = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)

            batch_x = tf.Variable(batch_x)
            batch_y = tf.Variable(batch_y)

            batch_y = tf.one_hot(batch_y, 9)
            with tf.GradientTape() as tape:
                logits = nn_model(batch_x, W1, b1, W2, b2)
                loss = loss_fn(logits, batch_y)

            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimalizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        test_logits = nn_model(x_test, W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")

    print("Traning completed")



if __name__ == '__main__':
    train()







    # for row in data:
    #     print(' '.join('{:3}'.format(value) for value in row))
    # (x_test, y_test) = load_dataset(TEST_DIR)
    # (x_train, y_train) = load_dataset(TRAIN_DIR)
    # print(x_test)
