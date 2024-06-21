import torch
from model import ViolenceClassifier  # 导入模型定义
import sys
import os
from PIL import Image
from torchvision import transforms


class ViolenceClass:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_model(self, model_path):
        model = ViolenceClassifier(num_classes=2)  # 初始化模型
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading the model: {e}")
            sys.exit(1)
        return model

    def classify(self, input_tensor: torch.Tensor) -> list:
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().tolist()

    def classify_folder(self, folder_path: str) -> list:
        image_list = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image = Image.open(file_path).convert('RGB')  # 确保图像是RGB格式
                image_tensor = self.preprocess(image)
                image_list.append(image_tensor)
        if not image_list:
            print("No images found in the folder.")
            return []

        input_tensor = torch.stack(image_list)  # 转换为 n*3*224*224 的tensor
        return self.classify(input_tensor)


# 示例用法
if __name__ == "__main__":
    # 假设模型文件路径
    model_path = "/your/path/to/model/"

    # 初始化分类器
    classifier = ViolenceClass(model_path)

    # 进行文件夹图像预测的示例
    folder_path = "/your/path/to/folder/"
    folder_predictions = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')
