from generate_expr import GenerateExpressionDataset, LABEL_TO_ID, ID_TO_CHAR, NUM_CLASSES, DATASET_PATH
from mathsolver import MathSolverModel
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from early_stopping_pytorch import EarlyStopping
import matplotlib.pyplot as plt

import numpy as np
from glob import glob
from typing import List, Tuple
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    input_lengths = torch.full((len(labels),), imgs.shape[-1] // 4, dtype=torch.long)  # Width/4 after pooling
    labels = torch.cat(labels)
    return imgs, labels, input_lengths, label_lengths

train_path = os.path.normpath(DATASET_PATH)
label_dir = train_path + "/train/"
# label_dir = "../maths-dataset/train"

def preprocess_generate_expression_dataset(num_samples=10000, batch_size=64):
    transform = transforms.Compose([
        # transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = GenerateExpressionDataset(label_dir=label_dir, num_samples=num_samples, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, val_loader

def get_four_sequence_images(train_loader):
    dataiter = iter(train_loader)
    imgs, labels, input_lengths, label_lengths = next(dataiter)

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # imshow(torchvision.utils.make_grid(imgs[:4]))

    # print(' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(4)))

def train(train_loader, val_loader, num_epochs=10, device=device, lr=1e-3, step_size=10, gamma=0.1, patience=5):
    model = MathSolverModel(NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=NUM_CLASSES - 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    epoch_vs_loss = {}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_vs_loss[epoch] = 0.0
        for imgs, labels, input_lengths, label_lengths in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)  # (width, batch, channel)
            loss = criterion(output, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        epoch_vs_loss[epoch] = loss

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for imgs, labels, input_lengths, label_lengths in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                output = model(imgs)
                loss = criterion(output, labels, input_lengths, label_lengths)
                val_loss += loss.item()
            print(f"Validation Loss = {val_loss / len(val_loader):.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        scheduler.step()

    torch.save(model.state_dict(), 'math_solver.pth')

    return epoch_vs_loss


def decode_prediction(pred):
    pred = pred.argmax(dim=2)  # (W, B)
    pred = pred.permute(1, 0)  # (B, W)
    decoded = []
    for seq in pred:
        expr = []
        prev = -1
        for p in seq:
            if p.item() != prev and p.item() != NUM_CLASSES - 1:
                expr.append(ID_TO_CHAR.get(p.item(), '?'))
            prev = p.item()
        decoded.append("".join(expr))
    return decoded

def run_inference(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (imgs, labels, input_lengths, label_lengths) in enumerate(data_loader):
            imgs = imgs.to(device)
            output = model(imgs)
            decoded = decode_prediction(output)

            label_ptr = 0
            for j in range(imgs.size(0)):
                label_len = label_lengths[j].item()
                gt = [ID_TO_CHAR[i.item()] for i in labels[label_ptr:label_ptr + label_len]]
                label_ptr += label_len

                gt_expr = ''.join(gt)
                print(f"GT: {gt_expr} | Pred: {decoded[j]}")
                if decoded[j] == gt_expr:
                    correct += 1
                total += 1

                if i < 5 and j == 0:
                    img_vis = to_pil_image(imgs[j].cpu() * 0.5 + 0.5)
                    img_vis.show(title=f"GT: {gt_expr}, Pred: {decoded[j]}")

    print(f"Test Accuracy: {correct / total * 100:.2f}%")


def infer_single_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]
    # plot the image
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = decode_prediction(output)
    
    print(f"Predicted Expression: {pred}")
    return pred[0]