import struct
from pathlib import Path
import numpy as np

DATA_DIR = Path(r"E:\深度学习\deeplearn\data\mnist")
TRAIN_IMAGES = DATA_DIR / "train-images-idx3-ubyte" / "train-images.idx3-ubyte"
TRAIN_LABELS = DATA_DIR / "train-labels-idx1-ubyte" / "train-labels.idx1-ubyte"
TEST_IMAGES = DATA_DIR / "t10k-images-idx3-ubyte" / "t10k-images.idx3-ubyte"
TEST_LABELS = DATA_DIR / "t10k-labels-idx1-ubyte" / "t10k-labels.idx1-ubyte"


def load_images(path):
    with open(path, "rb") as file:
        _, count, rows, cols = struct.unpack(">IIII", file.read(16))
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255

def load_labels(path):
    with open(path, "rb") as file:
        file.read(8)
        return np.frombuffer(file.read(), dtype=np.uint8).astype(np.int64)


class MLP:
    def __init__(self, layer_num, in_dims, out_dims, seed=0):
        self.layer_num = layer_num
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for i in range(layer_num):
            if i > 0 and in_dims[i] != out_dims[i - 1]:
                raise ValueError("相邻层维度不匹配")
            weight = rng.standard_normal((in_dims[i], out_dims[i]))
            weight *= np.sqrt(2 / in_dims[i])
            self.weights.append(weight.astype(np.float32))
            self.biases.append(np.zeros(out_dims[i], dtype=np.float32))

    @staticmethod
    def softmax(x):
        x -= x.max(axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, x):
        outputs = [x]
        before_relu = []

        for i in range(self.layer_num):
            z = outputs[-1] @ self.weights[i] + self.biases[i]
            before_relu.append(z)
            outputs.append(self.softmax(z) if i == self.layer_num - 1 else np.maximum(z, 0))

        return outputs[-1], outputs, before_relu

    def train_batch(self, x, y, learning_rate):
        probabilities, outputs, before_relu = self.forward(x)
        batch_size = len(y)

        loss = -np.log(probabilities[np.arange(batch_size), y] + 1e-12).mean()
        delta = probabilities.copy()
        delta[np.arange(batch_size), y] -= 1
        delta /= batch_size

        for i in range(self.layer_num - 1, -1, -1):
            weight_gradient = outputs[i].T @ delta
            bias_gradient = delta.sum(axis=0)

            if i > 0:
                delta = delta @ self.weights[i].T
                delta[before_relu[i - 1] <= 0] = 0
            self.weights[i] -= learning_rate * weight_gradient
            self.biases[i] -= learning_rate * bias_gradient

        return loss

    def predict(self, x, batch_size=1024):
        result = []
        for start in range(0, len(x), batch_size):
            probabilities, _, _ = self.forward(x[start:start + batch_size])
            result.append(probabilities.argmax(axis=1))
        return np.concatenate(result)

def main():
    print("正在读取 MNIST 数据...")

    # 读取真实的 MNIST 数据
    train_x = load_images(TRAIN_IMAGES)
    train_y = load_labels(TRAIN_LABELS)
    test_x = load_images(TEST_IMAGES)
    test_y = load_labels(TEST_LABELS)

    print("训练集：", train_x.shape, train_y.shape)
    print("测试集：", test_x.shape, test_y.shape)

    model = MLP(2, [784, 128], [128, 10])

    epochs = 5
    batch_size = 128
    learning_rate = 0.1
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        train_x, train_y = train_x[order], train_y[order]
        losses = []

        for start in range(0, len(train_x), batch_size):
            end = start + batch_size
            losses.append(model.train_batch(
                train_x[start:end],
                train_y[start:end],
                learning_rate
            ))

        accuracy = np.mean(model.predict(test_x) == test_y)
        print(f"第 {epoch + 1}/{epochs} 轮：损失 = {np.mean(losses):.4f}，测试准确率 = {accuracy:.2%}")

if __name__ == "__main__":
    main()
