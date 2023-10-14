import idx2numpy
import numpy as np

# 加载数据集
train_images = idx2numpy.convert_from_file("./M/t10k-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("./M/t10k-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("./M/train-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("./M/train-labels.idx1-ubyte")

# 将图像数据重塑为一维数组
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# 定义激活函数和softmax
def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()

# 初始化参数
def init_parameters():
    dimes = [28 * 28, 10]
    parameters = [
        {'b': np.zeros(dimes[0])},
        {'b': np.zeros(dimes[1]), 'w': np.random.rand(dimes[0], dimes[1])}
    ]
    return parameters

# 预测函数
def predict(img, parameters):
    I0_in = img + parameters[0]['b']
    I0_out = tanh(I0_in)
    I1_in = np.dot(I0_out, parameters[1]['w']) + parameters[1]['b']
    I1_out = softmax(I1_in)
    return I1_out

# 交叉熵损失函数
def cross_entropy_loss(prediction, label):
    return -np.log(prediction[label])

# 梯度下降训练
def gradient_descent(parameters, input_data, labels, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(input_data)):
            img = input_data[i]
            label = labels[i]

            # 前向传播
            I0_in = img + parameters[0]['b']
            I0_out = tanh(I0_in)
            I1_in = np.dot(I0_out, parameters[1]['w']) + parameters[1]['b']
            I1_out = softmax(I1_in)

            # 计算损失
            loss = cross_entropy_loss(I1_out, label)
            total_loss += loss

            # 反向传播更新参数
            grad_I1_out = I1_out.copy()
            grad_I1_out[label] -= 1
            grad_I0_out = np.dot(grad_I1_out, parameters[1]['w'].T)
            grad_I0_in = (1 - np.square(I0_out)) * grad_I0_out

            parameters[1]['w'] -= learning_rate * np.outer(I0_out, grad_I1_out)
            parameters[1]['b'] -= learning_rate * grad_I1_out
            parameters[0]['b'] -= learning_rate * grad_I0_in

        # 输出每轮的平均损失
        avg_loss = total_loss / len(input_data)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

# 初始化参数
parameters = init_parameters()

# 设置超参数
learning_rate = 0.00014
epochs = 640

# 进行梯度下降训练
gradient_descent(parameters, train_images, train_labels, learning_rate, epochs)

# 测试模型
correct_predictions = 0
for i in range(len(test_images)):
    img = test_images[i]
    label = test_labels[i]
    prediction = predict(img, parameters)
    if prediction.argmax() == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_images)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

parameters = [
    {'b': np.zeros(784)},
    {'b': np.zeros(10), 'w': np.random.rand(784, 10)}
]

# 保存模型参数到文件
np.save("../model_parameters.npy", parameters)
