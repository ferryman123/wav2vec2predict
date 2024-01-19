# -*- coding: utf-8 -*-

from transformers import AutoProcessor, Wav2Vec2Model
import torch
from sklearn.model_selection import train_test_split
import librosa
import json
import os

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def process_folder(folder_path, model_name="facebook/wav2vec2-base-960h"):
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

    # Get a list of all .wav files in the folder
    wav_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".wav")]

    # Initialize an empty list to store last_hidden_states
    all_hidden_states = []

    eyeBlinkLeft_list = []

    json_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".json")]

    # Process each .wav file in the folder
    for i in range(len(wav_files)):
        # Load audio file
        audio_input, rate = librosa.load(os.path.join(folder_path, wav_files[i]), sr=16000)

        # Process audio input with the processor
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the last hidden states
        last_hidden_states = outputs.last_hidden_state

        with open(os.path.join(folder_path, json_files[i]), 'r') as file:
            # Load the JSON data
            data = json.load(file)

            # Check if "blendshapes" key is present in the JSON
            if "blendshapes" in data:
                # Check if "jawOpen" key is present in the blendshapes
                if "eyeBlinkLeft" in data["blendshapes"]:
                    # Append the "jawOpen" values to the list

                    eyeBlinkLeft_list.extend(data["blendshapes"]["eyeBlinkLeft"])
                    n = len(data["blendshapes"]["eyeBlinkLeft"])
        # 使用切片进行降采样
        downsampled_last_hidden_states = last_hidden_states[:, ::2, :]


        # 删除最后len(last_hidden_states)-n个元素
        downsampled_last_hidden_states = downsampled_last_hidden_states[:, :-(len(last_hidden_states)//2-n), :]

        # Append to the list
        all_hidden_states.append(downsampled_last_hidden_states)

    return all_hidden_states,eyeBlinkLeft_list
def normalize_hidden_states(hidden_states_list):
    normalized_states_list = []

    for last_hidden_states in hidden_states_list:
        # Find min and max values along the second dimension (dim=1)
        min_values, _ = torch.min(last_hidden_states, dim=1, keepdim=True)
        max_values, _ = torch.max(last_hidden_states, dim=1, keepdim=True)

        # Normalize the hidden states
        normalized_states = (last_hidden_states - min_values) / (max_values - min_values)

        # Append to the list
        normalized_states_list.append(normalized_states)

    return normalized_states_list


path = "final"

hidden_states_list,eyeBlinkLeft = process_folder(path)
normalized_hidden_states_list = torch.cat(normalize_hidden_states(hidden_states_list),dim=1)

print(eyeBlinkLeft)
print(normalized_hidden_states_list)

print(len(eyeBlinkLeft))

eyeBlinkLeft_tensor = torch.tensor(eyeBlinkLeft, dtype=torch.float32)


# 重塑 x 和 y 的形状以适应线性回归
x_reshaped = normalized_hidden_states_list.reshape(-1, 768)
y_reshaped = eyeBlinkLeft_tensor.reshape(-1, 1)

print(x_reshaped.shape)

print(y_reshaped.shape)

# 划分数据集为训练集和测试集和验证集
# 划分数据为训练集（80%）和临时集（20%）
x_train, x_temp, y_train, y_temp = train_test_split(x_reshaped, y_reshaped, test_size=0.2, random_state=42)

# 再次划分临时集为验证集（50%）和测试集（50%）
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 现在 x_train, x_val, x_test 是训练集、验证集和测试集，y_train, y_val, y_test 是相应的标签

# 训练集
print(x_train.shape, y_train.shape)
# 验证集
print(x_val.shape, y_val.shape)
# 测试集
print(x_test.shape, y_test.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

# 创建 DataLoader 对象
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 定义深层神经网络模型
class DeepRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepRegressionModel, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 实例化神经网络模型
input_size = X_train_tensor.shape[1]
hidden_sizes = [256, 128,64,32]
output_size = 1
model = DeepRegressionModel(input_size, hidden_sizes, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型预测
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# 转换为NumPy数组
y_pred = y_pred_tensor.cpu().numpy()

# 评估性能
mse_nn = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Neural Network): {mse_nn}")

torch.save(model.state_dict(), 'model.pt')

y_pred[y_pred<0] = 0

print(y_pred)
print(y_test)

from sklearn.metrics import r2_score

# 计算 R²
r_squared = r2_score(y_test, y_pred)

print(r_squared)