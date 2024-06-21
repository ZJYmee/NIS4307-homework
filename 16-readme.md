# 人工智能导论课程大作业 - 接口调用实例说明
本文档旨在为使用接口类 ViolenceClass 进行图像分类的操作提供指导。

## 环境配置

建议使用anaconda/miniconda进行环境管理（可选）。

版本要求如下：
```
python>3.0.0
pytorch<2.0.0
torchvision<0.14
```
推荐版本：
```
python==3.8
pytorch==1.8.2
torchvision==0.9.2
```
## 接口类说明
接口类 ViolenceClass 实现了一个图像分类器，用于将输入的图像数据进行分类预测。以下是接口类的主要功能和使用方法：

### 初始化
```
from classify import ViolenceClass

# 假设模型文件路径
model_path = "path_to_your_trained_model.pth"

# 初始化分类器
classifier = ViolenceClass(model_path)
```
### 图像分类
```
from PIL import Image
import torch

# 加载图像并进行预处理
img_path = "path_to_your_image.jpg"
image = Image.open(img_path)
image_tensor = torch.unsqueeze(transform(image), 0) # 转换为 1*3*224*224 的tensor

# 调用分类接口
predictions = classifier.classify(image_tensor)

# 打印分类结果
print(f"Predicted class: {predictions}")
```
### 文件夹图像分类
```
# 假设有一个文件夹存放多张待分类图像
folder_path = "path_to_your_image_folder"

# 调用分类接口进行批量分类
folder_predictions = classifier.classify_folder(folder_path)

# 打印文件夹中每张图像的预测结果
print(f"Folder image predictions: {folder_predictions}")
```
## 示例用法
```
if __name__ == "__main__":
    # 假设模型文件路径
    model_path = "E:\\人工智能导论\\大作业\\train_logs\\shufflenet_pretrain_test\\version_9\\checkpoints\\shufflenet_pretrain_test-epoch=07-val_loss=0.09.pth"

    # 初始化分类器
    classifier = ViolenceClass(model_path)

    # 进行文件夹图像预测的示例
    folder_path = "E:\\人工智能导论\\大作业\\violence_224\\aiimade\\"
    folder_predictions = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')
```
## 注意事项
1. 模型文件路径：确保提供正确的模型文件路径，以便加载训练好的模型进行分类。
2. 图像预处理：接口类会自动处理图像的加载和预处理，但请确保输入的图像符合尺寸要求标准为224*224 RGB格式。

以上是使用 ViolenceClass 接口类进行图像分类的基本说明。如有问题或需要进一步指导，请随时联系。
