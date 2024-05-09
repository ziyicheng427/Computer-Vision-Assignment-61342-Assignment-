import cv2
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
        if descriptors is None:
            descriptors = np.zeros((1, sift.descriptorSize()))
        descriptors_list.append(descriptors)
    return descriptors_list

def extract_harris_features(images):
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)
        # Extract keypoints for the Harris corners
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        descriptors = centroids.astype(np.float32)
        descriptors_list.append(descriptors)
    return descriptors_list

# 提取特征
train_sift_features = extract_sift_features(X_train)
test_sift_features = extract_sift_features(X_test)
train_harris_features = extract_harris_features(X_train)
test_harris_features = extract_harris_features(X_test)

# 函数，用于转换特征列表为 NumPy 数组
def prepare_features(feature_list):
    features = [f.mean(axis=0) if len(f) > 0 else np.zeros((1, f.shape[1])) for f in feature_list]
    return np.vstack(features)

# 特征处理
train_sift = prepare_features(train_sift_features)
test_sift = prepare_features(test_sift_features)
train_harris = prepare_features(train_harris_features)
test_harris = prepare_features(test_harris_features)

# 添加噪声函数
def add_noise(images):
    noise_factor = 0.2
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy_images, 0.0, 255.0).astype(np.uint8)

# 加入噪声到测试集
X_test_noisy = add_noise(X_test)

# 分类器列表
classifiers = {
    "SIFT+Random Forest": RandomForestClassifier(n_estimators=50),
    "SIFT+SVM": SVC(),
    "Harris+Random Forest": RandomForestClassifier(n_estimators=50),
    "Harris+SVM": SVC(),
    "Harris+KNN": KNeighborsClassifier(n_neighbors=5)
}

# 准备数据和标签
features = {
    "SIFT": (train_sift, test_sift),
    "Harris": (train_harris, test_harris)
}

# 训练和评估每个分类器
results = {}
results_noisy = {}
for name, classifier in classifiers.items():
    feature_type = name.split('+')[0]
    train_features, test_features = features[feature_type]

    # 训练模型
    start_time = time.time()
    classifier.fit(train_features, y_train)
    train_time = time.time() - start_time

    # 在原始测试集上评估
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(y_test, predictions)
    results[name] = (accuracy, train_time)
    
    # 在噪声测试集上评估
    if feature_type == 'SIFT':
        test_features_noisy = prepare_features(extract_sift_features(X_test_noisy))
    else:
        test_features_noisy = prepare_features(extract_harris_features(X_test_noisy))
    
    predictions_noisy = classifier.predict(test_features_noisy)
    accuracy_noisy = accuracy_score(y_test, predictions_noisy)
    results_noisy[name] = accuracy_noisy

# 输出结果
for method, (acc, time_taken) in results.items():
    print(f"{method} Accuracy: {acc:.4f}, Time: {time_taken:.2f} seconds")
    print(f"{method} Noisy Accuracy: {results_noisy[method]:.4f}")
