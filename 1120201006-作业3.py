# -*- coding = utf-8 -*-
# @Time:2021/5/17 17:30
# @Author:单枭峰
# @File:1120201006作业3.py
# @Software:PyCharm

import pandas as pd
import numpy as np
import matplotlib.cm as cm
import  matplotlib.pyplot as plt
import  tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 读取CSV文件，并返回一个数据帧（Dataframe）
train_df = pd.read_csv("C:/Users/Administrator/Desktop/sign_mnist_train.csv")
test_df = pd.read_csv("C:/Users/Administrator/Desktop/sign_mnist_test.csv")

# 探索数据
# train_df.head()

# 提取标签
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# 提取图像
x_train = train_df.values
x_test = test_df.values

# x_train.shape
# y_train.shape
# x_test.shape
# y_test.shape


# 数据可视化
plt.figure(figsize=(40, 40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]

    image = row.reshape(28, 28)
    plt.subplot(1, num_images, i + 1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')  # plt.imshow（）负责对图像进行处理，并显示其格式，但不能显示
    plt.show()

# 数据归一化
x_train = np.multiply(x_train,1.0/255.0)
x_test = np.multiply(x_test,1.0/255.0)



# 对标签多分类编码
num_classes = 24
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train.shape



# 构建模型
# 输入层，隐藏层，输出层
# 1，模型实例化
model = Sequential()
# 2，创建输入层
model.add(Dense(units=512,activation='relu',input_shape=(784,)))
# 3，创建隐藏层
model.add(Dense(units=512,activation='relu'))
# 4,创建输出层
model.add(Dense(units=24,activation='softmax'))


# 总结模型
model.summary()  # 用来打印出可读的模型摘要


# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# 训练模型
# fit方法需要的参数 （1，训练数据及其标签，2，训练次数epochs. 3，验证数据）
history = model.fit(x_train,y_train,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test,y_test))
