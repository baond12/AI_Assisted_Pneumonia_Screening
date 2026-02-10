import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import time
import copy

# --- 1. CẤU HÌNH & XỬ LÝ DỮ LIỆU ---
DATA_DIR = '/content/drive/MyDrive/Data_Mining_HK252/chest_xray'  # Sửa lại đường dẫn nếu cần
BATCH_SIZE = 32
IMG_SIZE = 224

# Data Augmentation cho Train, chỉ Normalize cho Val/Test
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Tăng cường dữ liệu
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_data(data_dir):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes # ['NORMAL', 'PNEUMONIA']
    
    return dataloaders, dataset_sizes, class_names

# --- 2. XÂY DỰNG MODEL (TRANSFER LEARNING) ---
def build_model(device):
    # Load ResNet18 đã pre-train
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze (đóng băng) các lớp convolution để giữ đặc trưng ảnh
    for param in model.parameters():
        param.requires_grad = False
        
    # Thay thế lớp Fully Connected cuối cùng
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Output 2 lớp: Normal, Pneumonia
    
    model = model.to(device)
    return model

# --- 3. QUÁ TRÌNH HUẤN LUYỆN (TRAIN & VAL) ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("Bắt đầu huấn luyện...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Mỗi epoch có pha Train và Val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass chỉ ở pha Train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy model nếu đạt accuracy tốt nhất trên tập Val
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("--> Đã lưu checkpoint mới tốt nhất!")

        print()

    time_elapsed = time.time() - since
    print(f'Huấn luyện hoàn tất trong {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # Load lại trọng số tốt nhất
    model.load_state_dict(best_model_wts)
    return model, history

# --- 4. ĐÁNH GIÁ ĐỘC LẬP (TEST SET) ---
def evaluate_test_set(model, dataloader, device, class_names):
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST (Dữ liệu chưa từng thấy) ---")
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # Báo cáo chi tiết
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Vẽ Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - TEST SET')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    dataloaders, dataset_sizes, class_names = load_data(DATA_DIR)
    
    # 2. Xây dựng lại kiến trúc model (Khung rỗng)
    model = build_model(device)
    
    # --- PHẦN QUAN TRỌNG: LOAD TRỌNG SỐ CŨ ---
    checkpoint_path = '/content/drive/MyDrive/Data_Mining_HK252/pneumonia_resnet18_best.pth' # Tên file bạn đã lưu ở 5 epoch trước
    
    if os.path.exists(checkpoint_path):
        print(f"--> Đang load trọng số từ: {checkpoint_path}")
        # Load state_dict vào model
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("--> Không tìm thấy file cũ, train từ đầu!")
        
    # 3. Thiết lập Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    
    # MẸO: Khi train tiếp (fine-tuning), nên giảm Learning Rate xuống nhỏ hơn (vd: 0.0001)
    # để model học tinh chỉnh nhẹ nhàng, tránh làm hỏng các đặc trưng đã học tốt trước đó.
    optimizer = optim.SGD(model.fc.parameters(), lr=0.0001, momentum=0.9) 
    
    # 4. Train thêm 50 epochs
    # Model lúc này đã có trí khôn của 5 epoch trước, nó sẽ học tiếp từ đó
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=50)
    
    # 5. Lưu lại kết quả mới
    torch.save(model.state_dict(), '/content/drive/MyDrive/Data_Mining_HK252/pneumonia_resnet18_final.pth')
    print("Đã hoàn tất train thêm 50 epochs!")