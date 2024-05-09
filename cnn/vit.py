import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision.models import vit_b_16

# Load and preprocess CIFAR-10 data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ViT input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the Vision Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, 10)  # Correctly access and modify the classifier
model = model.to(device)

# Set up training environment
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
train_acc_history = []
test_acc_history = []
for epoch in range(epochs):
    model.train()
    total_correct = 0
    total_images = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_images += labels.size(0)

    train_accuracy = total_correct / total_images
    train_acc_history.append(train_accuracy)
    
    model.eval()
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

    test_accuracy = total_correct / total_images
    test_acc_history.append(test_accuracy)

    print(f'Epoch {epoch+1}, Train Accuracy: {train_accuracy*100}%, Test Accuracy: {test_accuracy*100}%')

# Plot accuracy graph
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(test_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.show()
