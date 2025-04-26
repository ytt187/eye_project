# -*- coding: utf-8 -*-
"""
完整眼科图像分类系统（优化修正版）
主要改进：
1. 修正模型前向传播错误
2. 完善困难样本训练流程
3. 优化特征保存机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.cuda.amp

# ==================== 数据增强模块 ====================
class MedicalAugmentation:
    def __init__(self):
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
          #  transforms.RandomGrayscale(p=0.2),  # 新增
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 针对混淆类别的增强
        self.confused_augment = transforms.RandomChoice([
            transforms.Lambda(self.add_glaucoma_sim),
            transforms.Lambda(self.add_myopia_sim),
            transforms.Lambda(self.add_health_noise)
        ])
        
        # 完整训练增强流程
        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomAffine(15, translate=(0.1, 0.1))], p=0.5),
            self.confused_augment,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(self.add_medical_artifacts),
            self.base_transform
        ])

    def add_medical_artifacts(self, img):
        """添加医学伪影"""
        if random.random() > 0.6:
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            cv2.circle(img_np, (random.randint(0,w), random.randint(0,h)),
                       random.randint(2,5), (255,0,0), -1)
            if random.random() > 0.8:
                cv2.rectangle(img_np,
                            (random.randint(0,w-15), random.randint(0,h-15)),
                            (random.randint(5,15), random.randint(5,15)),
                            (255,255,0), -1)
            img = Image.fromarray(img_np.astype('uint8'))
        return img

    def add_glaucoma_sim(self, img):
        """模拟青光眼特征"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        cv2.ellipse(img_np, (w//2, h//2), (30, 20), 0, 0, 360, (180,180,180), -1)
        return Image.fromarray(img_np)

    def add_myopia_sim(self, img):
        """模拟近视特征"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        cv2.ellipse(img_np, (w//2+20, h//2), (40, 30), 30, 0, 360, (200,200,200), -1)
        return Image.fromarray(img_np)

    def add_health_noise(self, img):
        """健康眼噪声"""
        img_np = np.array(img)
        noise = np.random.normal(0, 10, img_np.shape).astype('uint8')
        return Image.fromarray(np.clip(img_np + noise, 0, 255))
    
    def add_hard_case_aug(self, img):
        img_np = np.array(img)
        h, w = img_np.shape[:2]
    
    # 随机选择增强类型
        aug_type = random.choice(['health', 'glaucoma', 'myopia'])
        if aug_type == 'glaucoma':
           cv2.circle(img_np, (w//2, h//2), 20, (180,180,180), -1)  # 杯凹
        elif aug_type == 'myopia':
           cv2.ellipse(img_np, (w//2+30, h//2), (40,30), 15, 0, 360, (200,200,200), -1)
        else:  # health
          cv2.GaussianBlur(img_np, (5,5), 1)  # 模拟正常变异
      
        return Image.fromarray(img_np)
        
    """针对健康/青光眼/近视的增强"""
    
    
    
    def __call__(self, img, mode='train'):
        return self.train_transform(img) if mode == 'train' else self.base_transform(img)

# ==================== 数据集模块 ====================
class MedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.classes = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {cls:i for i,cls in enumerate(self.classes)}
        
        samples = []
        for cls in self.classes:
            imgs = list((Path(root_dir)/cls).glob('*.[jJpP][pPnN][gG]'))
            samples.extend([(str(p), self.class_to_idx[cls]) for p in imgs])
        
        train_val, test = train_test_split(samples, test_size=0.1, stratify=[s[1] for s in samples])
        train, val = train_test_split(train_val, test_size=0.11, stratify=[s[1] for s in train_val])
        
        self.samples = train if mode=='train' else val if mode=='val' else test
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img, self.mode)
        return img, label

# ==================== 模型架构 ====================
class AttEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        self.save_features = False  # 特征保存开关
        
        # 冻结前50%层
        blocks = list(self.backbone.features.children())
        for i in range(len(blocks)//2):
            for param in blocks[i].parameters():
                param.requires_grad = False
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1536, 64, 1),
            nn.GELU(),
            nn.LayerNorm([64, 1, 1]),
            nn.Conv2d(64, 1536, 1),
            nn.Sigmoid()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 512),
            nn.SiLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        
        if self.save_features:
            self.feature_maps = x.detach()
            
        att = self.channel_att(x)
        x = x * att
        x = self.backbone.avgpool(x)
        return self.classifier(torch.flatten(x, 1))

class AttEfficientNetWithCAM(AttEfficientNet):
    """扩展原有模型实现CAM功能"""
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        # 注册hook获取特征图
        self.feature_blobs = []
        self.backbone.features[-1].register_forward_hook(self._hook_features)
        
    def _hook_features(self, module, input, output):
        self.feature_blobs.append(output.detach())
    
    def get_cam_weights(self, class_idx):
        """获取指定类别的权重"""
        return self.classifier[-1].weight[class_idx]
    
    def generate_cam(self, img_tensor, class_idx):
        """生成类激活图"""
        # 清空特征缓存
        self.feature_blobs = []
        
        # 前向传播
        with torch.no_grad():
            logits = self.forward(img_tensor)
        
        # 获取特征图和权重
        features = self.feature_blobs[0].squeeze()
        weights = self.get_cam_weights(class_idx)
        
        # 计算CAM
        cam = torch.matmul(weights, features.view(features.size(0), -1))
        cam = cam.view(features.shape[1:])
        cam = F.relu(cam).cpu().numpy()
        
        # 后处理
        cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)
        return cam    

# ==================== 训练工具 ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, gamma_per_class=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_per_class = gamma_per_class
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        gamma = self.gamma_per_class[targets] if self.gamma_per_class is not None else self.gamma
        focal_term = (1 - pt) ** gamma
        
        if self.alpha is not None:
            loss = self.alpha[targets] * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
            
        return loss.mean()

def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total_loss, correct = 0, 0
    hard_samples = []
    
    for inputs, labels in tqdm(loader, desc=f'Training Epoch {epoch}'):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 收集困难样本
            preds = outputs.argmax(dim=1)
            confused_mask = torch.isin(labels, torch.tensor([1,7,8]).cuda())
            wrong_mask = (preds != labels)
            hard_samples.extend(zip(inputs[confused_mask & wrong_mask], 
                                  labels[confused_mask & wrong_mask]))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        correct += (preds == labels).sum().item()
    
    # 每3个epoch微调困难样本
    if epoch % 3 == 0 and hard_samples:
        finetune_hard_samples(model, hard_samples, optimizer, criterion, scaler)
    
    return total_loss/len(loader), correct/len(loader.dataset)

def finetune_hard_samples(model, hard_samples, optimizer, criterion, scaler):
    """困难样本微调"""
    hard_inputs = torch.stack([x[0] for x in hard_samples])
    hard_labels = torch.stack([x[1] for x in hard_samples])
    hard_loader = DataLoader(
        list(zip(hard_inputs, hard_labels)),
        batch_size=16,
        shuffle=True
    )
    
    original_state = {
        'lr': optimizer.param_groups[0]['lr'],
        'requires_grad': [p.requires_grad for p in model.parameters()]
    }
    
    # 只训练分类头
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    optimizer.param_groups[0]['lr'] = original_state['lr'] * 2
    
    model.train()
    for inputs, labels in tqdm(hard_loader, desc='Hard Sample Finetuning'):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 恢复原始状态
    for param, req_grad in zip(model.parameters(), original_state['requires_grad']):
        param.requires_grad = req_grad
    optimizer.param_groups[0]['lr'] = original_state['lr']

def validate(model, loader, plot_cm=True):
    """验证函数"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            outputs = model(inputs.to('cuda'))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds, target_names=train_set.classes))
    
    if plot_cm:
        plt.figure(figsize=(10,8))
        sns.heatmap(confusion_matrix(all_labels, all_preds), 
                   annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    return np.mean(np.array(all_preds) == np.array(all_labels))

# ==================== 主流程 ====================
if __name__ == '__main__':
    # 初始化
    data_dir = "Augmented Dataset" # 修改为实际路径
    transform = MedicalAugmentation()
    
    # 数据集
    train_set = MedicalDataset(data_dir, transform, 'train')
    val_set = MedicalDataset(data_dir, transform, 'val')
    
    # 类平衡采样
    sample_weights = [3.0 if label in [1,7,8] else 1.0 
                     for _, label in train_set.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=32)
    
    # 模型与优化器
    model = AttEfficientNet(num_classes=len(train_set.classes)).to('cuda')
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # 损失函数
    class_counts = np.bincount([label for _,label in train_set.samples])
    class_weights = torch.tensor(1. / (class_counts + 1e-4)).float().to('cuda')
    gamma_per_class = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 1.0, 2.5, 2.5, 1.5]).to('cuda')
    criterion = FocalLoss(alpha=class_weights, gamma_per_class=gamma_per_class)
    
    # 训练循环
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0
    
    for epoch in range(10):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
        val_acc = validate(model, val_loader, plot_cm=(epoch==9))
        
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2%}, Val Acc={val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'newbest_model.pth')
            
            
            
         