import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from data_loader import get_patient_files, build_dataset
from model import CNN_LSTM


# ==============================
# SEED (STABILITY)
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==============================
# SELECT PATIENT
# ==============================
patient_id = "chb24"
files = get_patient_files(patient_id)

print(f"\nTraining for patient: {patient_id}")
print(f"Files found: {len(files)}")

if len(files) == 0:
    print("❌ No files found")
    exit()


# ==============================
# LOAD DATA
# ==============================
X, y = build_dataset(files)

# Normalize (important)
X = (X - X.mean()) / (X.std() + 1e-8)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)


# ==============================
# TRAIN / TEST SPLIT
# ==============================
generator = torch.Generator().manual_seed(42)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)


# ==============================
# MODEL
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


# ==============================
# TRAINING
# ==============================
epochs = 20
patience = 5
best_loss = float('inf')
counter = 0

train_losses = []
train_accs = []

print("\n🚀 Training started...\n")

for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(outputs, 1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    acc = 100 * correct / total

    train_losses.append(total_loss)
    train_accs.append(acc)

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

    # Early stopping
    if total_loss < best_loss:
        best_loss = total_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("⛔ Early stopping triggered")
        break


# ==============================
# LOAD BEST MODEL
# ==============================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


# ==============================
# TESTING
# ==============================
correct = 0
total = 0

all_preds = []
all_labels = []
probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)

        pred = torch.argmax(outputs, 1)
        prob = torch.softmax(outputs, dim=1)[:, 1]

        correct += (pred == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
        probs.extend(prob.cpu().numpy())

test_acc = 100 * correct / total

print("\n✅ FINAL TEST ACCURACY:", test_acc, "%")


# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Interictal", "Preictal"]
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()


# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(all_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


# ==============================
# LOSS & ACCURACY GRAPH
# ==============================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.grid()

plt.subplot(1,2,2)
plt.plot(train_accs)
plt.title("Training Accuracy")
plt.grid()

plt.show()


# ==============================
# SAVE FINAL MODEL
# ==============================
torch.save(model.state_dict(), f"model_{patient_id}.pth")
print("\n💾 Model saved")