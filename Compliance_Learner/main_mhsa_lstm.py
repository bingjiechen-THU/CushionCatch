import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import math

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定义collate函数来处理批次数据
def collate_fn(batch):
    inputs, targets, file_names = zip(*batch)
    
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.long)
    file_names = list(file_names)
    
    return padded_inputs, input_lengths, padded_targets, target_lengths, file_names

class BallDataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.file_list = file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, skiprows=1, header=None, dtype=float)
        
        data_velocity = data.iloc[:, :3]
        data_diff = data_velocity.diff().iloc[1:]
        data_diff_mean = data_diff.mean(axis=1)
        
        max_index = data_diff_mean.abs().idxmax()
        
        inputs = torch.tensor(data.iloc[:max_index, :].values, dtype=torch.float32)
        labels = torch.tensor(data.iloc[max_index:, :].values, dtype=torch.float32)
        
        file_name = os.path.basename(file_path)
        
        return inputs, labels, file_name

def create_dataloader(data_dir, batch_size=32, shuffle=True, test_size=0.2, random_state=42):
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=random_state)

    train_dataset = BallDataset(data_dir, train_files)
    test_dataset = BallDataset(data_dir, test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, test_dataloader

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size, num_layers=2):
        super(MultiHeadAttentionModel, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        attn_output = x
        for attn_layer in self.attention_layers:
            attn_output, _ = attn_layer(attn_output, attn_output, attn_output)
        
        output = self.fc_out(attn_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class LSTMModelWithPositionEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(LSTMModelWithPositionEncoding, self).__init__()
        self.positional_encoding = PositionalEncoding(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.positional_encoding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc_out(lstm_out)
        return output

# 训练模型并保存权重
def train_and_save_model(model, train_loader, criterion, optimizer, device, seq_len=1, num_epochs=20, save_path="model_weights.pth"):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, input_lengths, targets, target_lengths, file_names in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            current_input = inputs.clone()
            loss = 0
            # 动态确定max_length
            max_length = target_lengths.max().item() if seq_len == 0 else seq_len

            all_outputs = []

            for t in range(max_length):
                outputs = model(current_input)
                next_preds = []
                for i, length in enumerate(target_lengths):
                    if t < length:
                        next_pred = outputs[i, t, :]
                        next_preds.append(next_pred)
                    else:
                        next_preds.append(torch.zeros_like(outputs[i, t, :]))  # 填充
                next_preds = torch.stack(next_preds, dim=0)
                next_preds = next_preds.unsqueeze(1)
                current_input = torch.cat((current_input, next_preds), dim=1)
                all_outputs.append(next_preds)
            
            all_outputs = torch.cat(all_outputs, dim=1)

            # 计算损失
            mask = torch.arange(max_length, device=device).expand(len(target_lengths), max_length) < target_lengths.unsqueeze(1).to(device)
            mask = mask.unsqueeze(-1).expand(-1, -1, targets.size(-1))  # 调整掩码形状以匹配目标
            all_outputs = all_outputs.masked_select(mask).view(-1, targets.size(-1))
            try:
                targets = targets[:, :max_length, :].masked_select(mask).view(-1, targets.size(-1))
            except Exception as e:
                print(e)
                print(file_names)
                continue
            
            loss = criterion(all_outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')
    
    # 保存权重文件
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

# 测试模型
def test_model(model, test_loader, criterion, device, seq_len=1, load_path=None):
    if load_path:
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, input_lengths, targets, target_lengths, file_names in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            current_input = inputs.clone()
            loss = 0
            max_length = target_lengths.max().item() if seq_len == 0 else seq_len

            all_outputs = []

            for t in range(max_length):
                outputs = model(current_input)
                next_preds = []
                for i, length in enumerate(target_lengths):
                    if t < length:
                        next_pred = outputs[i, t, :]
                        next_preds.append(next_pred)
                    else:
                        next_preds.append(torch.zeros_like(outputs[i, t, :]))  # 填充
                next_preds = torch.stack(next_preds, dim=0)
                next_preds = next_preds.unsqueeze(1)
                current_input = torch.cat((current_input, next_preds), dim=1)
                all_outputs.append(next_preds)
            
            all_outputs = torch.cat(all_outputs, dim=1)

            mask = torch.arange(max_length, device=device).expand(len(target_lengths), max_length) < target_lengths.unsqueeze(1).to(device)
            mask = mask.unsqueeze(-1).expand(-1, -1, targets.size(-1))  # 调整掩码形状以匹配目标
            all_outputs = all_outputs.masked_select(mask).view(-1, targets.size(-1))
            
            try:
                targets = targets[:, :max_length, :].masked_select(mask).view(-1, targets.size(-1))
            except Exception as e:
                print(e)
                print(file_names)
                continue
            
            loss = criterion(all_outputs, targets)
            total_loss += loss.item()
    
    print(f'Test Loss: {total_loss/len(test_loader)}')
    return total_loss / len(test_loader)


if __name__ == '__main__':
    # 固定随机种子
    set_seed(42)
    num_epochs = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = 'data/data_collection'
    save_path = 'ckpt/model_weights.pth'
    batch_size = 12
    train_dataloader, test_dataloader = create_dataloader(data_dir, batch_size=batch_size)

    input_size = 6  # 输入特征的维度
    hidden_size = 64  # 注意力层的隐藏层大小
    output_size = 6  # 输出特征的维度
    num_heads = 6  # 多头注意力中的头数量
    model = LSTMModelWithPositionEncoding(input_size, hidden_size, output_size, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练并保存模型
    train_and_save_model(model, train_dataloader, criterion, optimizer, device, seq_len=0, num_epochs=num_epochs, save_path=save_path)
    test_model(model, test_dataloader, criterion, device, seq_len=0, load_path=save_path)
   
