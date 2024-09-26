import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=32, shuffle=False)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        print(labels)
        _, predicted = torch.max(outputs.logits, 1)  # 获取预测的标签
        total += labels.size(0)
        predicted=predicted//100
        print(predicted)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')