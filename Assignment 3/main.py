import numpy as np
from torchvision.datasets import MNIST
import time


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten() / 255.0,
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def activation(x):
    return np.maximum(0, x)


def activation_derivative(x):
    return (x > 0).astype(float)


def softmax(z):
    max_exp=np.max(np.exp(z))
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))/max_exp
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions + 1e-9), axis=1))


def init_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def backpropagate(X_batch, Y_batch, a1, Y_pred, W1, W2, b1, b2, lambda_reg, learning_rate):
    m = X_batch.shape[0]

    delta2 = Y_pred - Y_batch
    W2_grad = np.dot(a1.T, delta2) / m + lambda_reg * np.square(W2)
    b2_grad = np.sum(delta2, axis=0, keepdims=True) / m

    delta1 = np.dot(delta2, W2.T) * activation_derivative(a1)
    W1_grad = np.dot(X_batch.T, delta1) / m + lambda_reg * np.square(W1)
    b1_grad = np.sum(delta1, axis=0, keepdims=True) / m

    W1 -= learning_rate * W1_grad
    b1 -= learning_rate * b1_grad
    W2 -= learning_rate * W2_grad
    b2 -= learning_rate * b2_grad

    return W1, b1, W2, b2


def train_mlp(train_X, train_Y, test_X, test_Y, epochs=50, batch_size=100, learning_rate=0.01, lambda_reg=0.0001):
    input_size = train_X.shape[1]
    hidden_size = 100
    output_size = 10

    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        indices = np.random.permutation(train_X.shape[0])
        train_X, train_Y = train_X[indices], train_Y[indices]

        for start in range(0, train_X.shape[0], batch_size):
            end = start + batch_size
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]

            z1 = np.dot(X_batch, W1) + b1
            a1 = activation(z1)
            z2 = np.dot(a1, W2) + b2
            Y_pred = softmax(z2)

            W1, b1, W2, b2 = backpropagate(X_batch, Y_batch, a1, Y_pred, W1, W2, b1, b2, lambda_reg, learning_rate)

        z1_train = np.dot(train_X, W1) + b1
        a1_train = activation(z1_train)
        z2_train = np.dot(a1_train, W2) + b2
        Y_train_pred = softmax(z2_train)
        loss_train = cross_entropy_loss(Y_train_pred, train_Y)

        z1_test = np.dot(test_X, W1) + b1
        a1_test = activation(z1_test)
        z2_test = np.dot(a1_test, W2) + b2
        Y_test_pred = softmax(z2_test)
        test_accuracy = np.mean(np.argmax(Y_test_pred, axis=1) == np.argmax(test_Y, axis=1))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_train:.4f} - Test Accuracy: {test_accuracy:.4f}")

    return W1, b1, W2, b2

start_time = time.time()

train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)

train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)

W1, b1, W2, b2 = train_mlp(train_X, train_Y_one_hot, test_X, test_Y_one_hot, epochs=70, batch_size=100,
                           learning_rate=0.01, lambda_reg=0.0001)

end_time = time.time()
elapsed_time = end_time - start_time
minutes = elapsed_time // 60
seconds = elapsed_time % 60
print(f"Training completed in {int(minutes)}m{int(seconds)}s")