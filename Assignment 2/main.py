import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./ data',
                    transform = lambda x: np.array(x).flatten(),
                    download = True,
                    train = is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def softmax(weighted_sum):
    exp_weighted_sum = np.exp(weighted_sum - np.max(weighted_sum, axis=1, keepdims=True))
    return exp_weighted_sum / np.sum(exp_weighted_sum, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions + 1e-9), axis=1))


def update_weights(X, Target, Y_pred, W, b, learning_rate=0.01):
    err = Target - Y_pred
    W += learning_rate * np.dot(X.T, err)
    b += learning_rate * np.mean(err, axis=0)
    return W, b


np.random.seed(212)
W = np.random.randn(784, 10)
b = np.zeros(10)


def train_perceptron(train_X, train_Y, test_X, test_Y, epochs=50, batch_size=100, learning_rate=0.01):
    global W, b
    train_size = train_X.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(train_size)
        train_X, train_Y = train_X[indices], train_Y[indices]

        for start in range(0, train_size, batch_size):
            end = start + batch_size
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]

            weighted_sum = np.dot(X_batch, W) + b
            Y_pred = softmax(weighted_sum)

            W, b = update_weights(X_batch, Y_batch, Y_pred, W, b, learning_rate)

        weighted_sum_train = np.dot(train_X, W) + b
        Y_train_pred = softmax(weighted_sum_train)
        loss_train = cross_entropy_loss(Y_train_pred, train_Y)

        weighted_sum_test = np.dot(test_X, W) + b
        Y_test_pred = softmax(weighted_sum_test)
        test_accuracy = np.mean(np.argmax(Y_test_pred, axis=1) == np.argmax(test_Y, axis=1))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_train:.4f} - Test Accuracy: {test_accuracy:.4f}")


train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)
print("Shape of train_Y_one_hot:", train_Y_one_hot.shape)
print("Shape of test_Y_one_hot:", test_Y_one_hot.shape)
train_perceptron(train_X, train_Y_one_hot, test_X, test_Y_one_hot, epochs=100, batch_size=500, learning_rate=0.01)



