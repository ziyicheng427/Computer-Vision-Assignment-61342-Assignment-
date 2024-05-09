import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist

def add_noise(images, noise_factor=0.8):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0, 255).astype(np.uint8)

# 加载 Fashion MNIST 数据集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 选择七个类别
classes_to_use = [0, 1, 2, 3, 4, 5, 6]
train_mask = np.isin(y_train, classes_to_use)
test_mask = np.isin(y_test, classes_to_use)

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# 扩展到三通道
x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)

# 分割数据集为 80% 训练集和 20% 测试集
X_train, X_test, y_train, y_test = train_test_split(np.concatenate((x_train, x_test), axis=0),
                                                    np.concatenate((y_train, y_test), axis=0),
                                                    test_size=0.20, random_state=42)

# SIFT 特征提取
sift = cv2.SIFT_create()

def extract_sift_features(images):
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.zeros((1, sift.descriptorSize()))
        descriptors_list.append(descriptors)
    return descriptors_list

train_descriptors = extract_sift_features(X_train)
test_descriptors = extract_sift_features(X_test)

train_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in train_descriptors if desc is not None])
test_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in test_descriptors if desc is not None])

# KNN 训练和测试
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_descriptors_stack, y_train)

# 原始数据准确率
predictions = knn.predict(test_descriptors_stack)
accuracy_original = accuracy_score(y_test, predictions)
print("原始数据准确率:", accuracy_original)

# 加入噪声后的准确率
X_test_noisy = add_noise(X_test)
test_descriptors_noisy = extract_sift_features(X_test_noisy)
test_descriptors_stack_noisy = np.array([np.mean(desc, axis=0) for desc in test_descriptors_noisy if desc is not None])
predictions_noisy = knn.predict(test_descriptors_stack_noisy)
accuracy_noisy = accuracy_score(y_test, predictions_noisy)
print("加噪声后的准确率:", accuracy_noisy)

# 加载全部十个类别的数据
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(np.concatenate((x_train, x_test), axis=0),
                                                    np.concatenate((y_train, y_test), axis=0),
                                                    test_size=0.20, random_state=42)

train_descriptors = extract_sift_features(X_train)
test_descriptors = extract_sift_features(X_test)

train_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in train_descriptors if desc is not None])
test_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in test_descriptors if desc is not None])

knn.fit(train_descriptors_stack, y_train)

# 全部十个类别的准确率
predictions_full = knn.predict(test_descriptors_stack)
accuracy_full = accuracy_score(y_test, predictions_full)
print("十个类别的准确率:", accuracy_full)
