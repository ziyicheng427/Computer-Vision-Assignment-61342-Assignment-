import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 合并数据集以便后续分割
X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0).flatten()

# 分割数据集为 80% 训练集和 20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 初始化 SIFT 特征提取器
sift = cv2.SIFT_create()

def extract_sift_features(images):
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        # 如果图像没有关键点，用全零填充描述符
        if descriptors is None:
            descriptors = np.zeros((1, sift.descriptorSize()))
        descriptors_list.append(descriptors)
    return descriptors_list

# 提取训练和测试数据的 SIFT 特征
train_descriptors = extract_sift_features(X_train)
test_descriptors = extract_sift_features(X_test)

# 将所有训练样本的 SIFT 描述符平均化以简化 KNN 训练
train_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in train_descriptors if desc is not None])
test_descriptors_stack = np.array([np.mean(desc, axis=0) for desc in test_descriptors if desc is not None])

# 训练 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_descriptors_stack, y_train)

# 进行预测
predictions = knn.predict(test_descriptors_stack)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("准确率:", accuracy)
