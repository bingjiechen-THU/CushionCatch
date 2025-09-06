import torch
import numpy as np
from compliance_learning.main_mhsa_lstm import LSTMModelWithPositionEncoding,set_seed

class POC_Traj_Learner():
    def __init__(self, load_path="./ckpt/model_weights.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 6  # 输入特征的维度
        hidden_size = 64  # 注意力层的隐藏层大小
        output_size = 6  # 输出特征的维度
        self.model = LSTMModelWithPositionEncoding(input_size, hidden_size, output_size, num_layers=2).to(device)
        
        # 加载模型权重
        self.load_model_weights(load_path, device)
    
    def load_model_weights(self, load_path, device):
        self.model.load_state_dict(torch.load(load_path))
        self.model.to(device)
        self.model.eval()  # 切换到评估模式
        print(f"Model weights loaded from {load_path}")

    def inference(self, inputs, device, max_length=100):
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)
            
            # 初始化当前输入为原始输入
            current_input = inputs.clone()
            
            # 循环生成预测序列直到达到最大长度
            for t in range(max_length):
                # 通过模型生成下一个时间步的预测
                outputs = self.model(current_input)
                next_pred = outputs[:, -1, :]  # 获取最后一个时间步的预测
                
                # 将预测结果添加到序列中
                next_pred = next_pred.unsqueeze(1)
                current_input = torch.cat((current_input, next_pred), dim=1)
                
                # 保存预测结果
                predictions.append(next_pred.cpu().numpy())
        
        print("Inference complete. Predictions generated.")
        predictions = np.concatenate(predictions, axis=1)
        return predictions

    def predict_vel(self, vel_posi_input):
        set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_length = 16  # 序列的最大长度
        input_size = 6  # 输入特征的维度
        # 输入序列,必须含有batch维度，输出也带有banch维度
        inputs = np.array(vel_posi_input).reshape(1, 1, input_size)
        
        # 输出maxlength段轨迹（xyz速度和位置）
        predictions = self.inference(inputs, device, max_length=max_length)
        # 去除batch维度
        predictions = predictions[0]
        # 只保留速度输出
        return predictions[:, :3]

