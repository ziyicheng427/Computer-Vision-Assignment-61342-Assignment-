import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 由于 ResNet50 要求的输入尺寸至少为 32x32，我们将图像尺寸调整至 64x64
x_train_resized = tf.image.resize(x_train, (64, 64))
x_test_resized = tf.image.resize(x_test, (64, 64))

# 载入 ResNet50 模型，不包括顶层
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# 冻结预训练模型的层
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)

# 创建最终模型
model = Model(inputs=base_model.input, outputs=x)

# 编译模型
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history_cifar = model.fit(x_train_resized, y_train, batch_size=64, epochs=30, validation_data=(x_test_resized, y_test), verbose=1)

# 评估模型
test_loss, test_accuracy = model.evaluate(x_test_resized, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy:.4f}")

# 显示模型结构
model.summary()



# Plotting the training results
plt.figure(figsize=(8, 6))
plt.plot(history_cifar.history['accuracy'], label='Training Accuracy')
plt.plot(history_cifar.history['val_accuracy'], label='Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy(ResNet50)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

