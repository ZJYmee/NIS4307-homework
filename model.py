import torch
from torch import nn
import numpy as np
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=6e-5, dropout_rate=0.3):
        super().__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True) # 加载预训练的 ShuffleNet v2 x1 模型
        num_ftrs = self.model.fc.in_features
        # 替换原来的全连接层
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),  # 假设将维度减半
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout 层
            nn.Linear(num_ftrs // 2, num_classes)  # 最后的分类层
        )
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch):
        # 显式地将模型设置为评估模式
        self.model.eval()
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def test_step(self, batch):
        self.model.eval()
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc, 'logits': logits, 'y': y}

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs])
        y_true = torch.cat([x['y'] for x in outputs])
        y_pred = torch.argmax(logits, dim=1)
        y_score = torch.softmax(logits, dim=1).cpu().numpy()

        self.analyze_results(y_true.cpu().numpy(), y_pred.cpu().numpy(), y_score)

    def analyze_results(self, y_true, y_pred, y_score):
        self.plot_confusion_matrix(y_true, y_pred) # 生成混淆矩阵
        self.plot_roc_curve(y_true, y_score) # 生成ROC曲线
        self.plot_precision_recall_curve(y_true, y_score) # 生成精确度-召回率曲线

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_roc_curve(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
