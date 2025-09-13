import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from medmnist import INFO, PathMNIST
from captum.attr import IntegratedGradients
import sqlite3
import io
import base64
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset & Transforms
# -----------------------------
data_flag = 'pathmnist'
info = INFO[data_flag]
DataClass = PathMNIST
n_classes = len(info['label'])

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=transform, download=True)
test_dataset = DataClass(split='test', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
]), download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -----------------------------
# Improved CNN Model
# -----------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Initialize model
model = ImprovedCNN(num_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

# -----------------------------
# Training Loop with Early Stopping
# -----------------------------
best_acc = 0
patience = 4
trigger_times = 0

for epoch in range(25):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    scheduler.step(accuracy)
    print(f"Epoch {epoch+1} — Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    if accuracy > best_acc:
        best_acc = accuracy
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# -----------------------------
# Evaluation on Test Set
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -----------------------------
# SQLite DB Setup
# -----------------------------
conn = sqlite3.connect("pathmnist_explanations.db")
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_index INTEGER UNIQUE,
    true_label INTEGER,
    predicted_label INTEGER,
    confidence REAL,
    attribution TEXT
)
''')
conn.commit()

def save_attribution_to_db(index, true_label, predicted_label, confidence, attribution):
    cursor.execute("SELECT 1 FROM explanations WHERE image_index=?", (index,))
    if cursor.fetchone():
        return

    buffer = io.BytesIO()
    np.save(buffer, attribution)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')

    cursor.execute("""
    INSERT INTO explanations (image_index, true_label, predicted_label, confidence, attribution)
    VALUES (?, ?, ?, ?, ?)
    """, (index, true_label, predicted_label, confidence, encoded))
    conn.commit()

# -----------------------------
# Integrated Gradients + DB
# -----------------------------
def visualize_and_store_ig(model, image, true_label, image_index, show=False):
    model.eval()

    def forward_func(x):
        return model(x)

    ig = IntegratedGradients(forward_func)
    output = model(image)
    probs = torch.softmax(output, dim=1)
    confidence, predicted_label = torch.max(probs, dim=1)

    attributions, _ = ig.attribute(image, target=predicted_label.item(), return_convergence_delta=True)
    attr_np = attributions.squeeze().cpu().detach().numpy()

    save_attribution_to_db(
        index=image_index,
        true_label=true_label,
        predicted_label=predicted_label.item(),
        confidence=confidence.item(),
        attribution=attr_np
    )

    if show:
        image_np = image.squeeze().cpu().detach().numpy()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(np.transpose(image_np, (1, 2, 0)))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Attribution Map")
        attr_norm = np.abs(attr_np).sum(axis=0)
        attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)
        plt.imshow(attr_norm, cmap='hot')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        overlay = np.transpose(image_np, (1, 2, 0))
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        heatmap = plt.cm.inferno(attr_norm)[..., :3]
        plt.imshow(overlay * 0.5 + heatmap * 0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# -----------------------------
# Process 500 Test Samples
# -----------------------------
print("\nProcessing 500 test samples with attribution + DB store...")
for i in tqdm(range(500)):
    image, label = test_dataset[i]
    image = image.unsqueeze(0).to(device)
    visualize_and_store_ig(model, image, label.item(), image_index=i, show=False)

# -----------------------------
# Display 2 Example Visualizations
# -----------------------------
print("\nDisplaying 2 sample visualizations from test set...")
for i in [10, 20]:
    image, label = test_dataset[i]
    image = image.unsqueeze(0).to(device)
    visualize_and_store_ig(model, image, label.item(), image_index=i, show=True)

# -----------------------------
# Print DB Summary
# -----------------------------
cursor.execute("SELECT COUNT(*) FROM explanations")
count = cursor.fetchone()[0]
print(f"\n Total entries in DB: {count}")
