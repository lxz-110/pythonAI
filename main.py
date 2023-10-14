import math
import tkinter as tk
import numpy as np

parameters = np.load("model/model_parameters89.14.npy", allow_pickle=True)


#神经网络
def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()


dimes = [28 * 28, 10]
activation = [tanh, softmax]
distutils = [
    {'b': [0, 0]},
    {'b': [0, 0], 'w': [-math.sqrt(6 / (dimes[0] + dimes[1])), math.sqrt(6 / (dimes[0] + dimes[1]))]},
]


def init_parameters_b(layer):
    dist = distutils[layer]['b']
    return np.random.rand(dimes[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters_w(layer):
    dist = distutils[layer]['w']
    return np.random.rand(dimes[layer - 1], dimes[layer]) * (dist[1] - dist[0]) + dist[0]


def init_parameters():#调试函数
    parameters = []
    for i in range(len(distutils)):
        layer_parameters = {}
        for j in distutils[i].keys():
            if j == 'b':
                layer_parameters['b'] = init_parameters_b(i)
            elif j == 'w':
                layer_parameters['w'] = init_parameters_w(i)
        parameters.append(layer_parameters)
    return parameters

def predict(img, parameters):
    I0_in = img + parameters[0]['b']
    I0_out = activation[0](I0_in)
    I1_in = np.dot(I0_out, parameters[1]['w']) + parameters[1]['b']
    I1_out = activation[1](I1_in)
    return I1_out

def clear_canvas(*canvas_list):
    for canvas in canvas_list:
        canvas.delete("all")
        # 如果是小画布，也清除小画布的内容
        if canvas == small_canvas:
            for y in range(28):
                for x in range(28):
                    small_canvas.create_rectangle(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill="black", outline="black")


def paint(event):
    x1, y1 = (event.x, event.y)
    x2, y2 = (event.x + 20, event.y + 20)
    large_canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

    # 在小Canvas上进行相同的绘制
    x1_small, y1_small = x1 // 10, y1 // 10
    x2_small, y2_small = x2 // 10, y2 // 10
    small_canvas.create_rectangle(x1_small, y1_small, x2_small, y2_small, fill="white", outline="white")


def predict_image():
    # 获取小Canvas上的图像数据
    image_data = []  # 存储图像数据的列表
    for y in range(28):
        row = []
        for x in range(28):
            pixel_color = small_canvas.itemcget(small_canvas.find_closest(x, y), 'fill')
            # 如果颜色为白色，将像素值设为1；否则设为0
            if pixel_color == "white":
                row.append(1)
            else:
                row.append(0)
        image_data.extend(row)

    # 将图像数据重塑为长度为 784 的一维数组
    img = np.array(image_data)

    # 使用神经网络进行预测
    prediction = predict(img, parameters)

    # 输出预测结果
    print("预测结果:", prediction)
    print("最大值索引:", prediction.argmax())




# 创建主窗口
root = tk.Tk()
root.title("绘制数字")

# 创建大的Canvas，大小为280x280
large_canvas = tk.Canvas(root, width=280, height=280, bg="black")
large_canvas.pack()

# 创建清除按钮
clear_button = tk.Button(root, text="清除全部", command=lambda: clear_canvas(small_canvas, large_canvas))
clear_button.pack()



# 创建确定按钮
predict_button = tk.Button(root, text="确定", command=predict_image)
predict_button.pack()

# 创建原始Canvas，大小为28x28
small_canvas = tk.Canvas(root, width=28, height=28, bg="black")
small_canvas.place(x=10, y=10)  # 将小Canvas放置在大Canvas上

# 绑定鼠标事件到大的Canvas
large_canvas.bind("<B1-Motion>", paint)

# 运行主循环
root.mainloop()
