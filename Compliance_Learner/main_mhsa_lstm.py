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


def set_seed(seed):
    """
    Set random seeds for reproducibility across different libraries.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    Custom collate function to handle padding for batches of variable-length sequences.

    Args:
        batch (list): A list of tuples, where each tuple contains (inputs, targets, file_name).

    Returns:
        tuple: A tuple containing padded inputs, input lengths, padded targets,
               target lengths, and a list of file names.
    """
    inputs, targets, file_names = zip(*batch)

    # Pad sequences to the length of the longest sequence in the batch
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Get the original lengths of the sequences before padding
    input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    target_lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.long)
    file_names = list(file_names)

    return padded_inputs, input_lengths, padded_targets, target_lengths, file_names


class BallDataset(Dataset):
    """
    Custom PyTorch Dataset for loading ball trajectory data.
    Each file is split into an input sequence and a target sequence based on
    the point of maximum velocity change.
    """

    def __init__(self, data_dir, file_list):
        """
        Initializes the dataset.

        Args:
            data_dir (str): The directory where the data files are stored.
            file_list (list): A list of filenames to be included in the dataset.
        """
        self.data_dir = data_dir
        self.file_list = file_list

    def __len__(self):
        """Returns the total number of files in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Args:
            idx (int): The index of the file to retrieve.

        Returns:
            tuple: A tuple containing the input tensor, label tensor, and file name.
        """
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, skiprows=1, header=None, dtype=float)

        # Calculate the point of maximum change in velocity to split the data
        data_velocity = data.iloc[:, :3]
        data_diff = data_velocity.diff().iloc[1:]
        data_diff_mean = data_diff.mean(axis=1)
        max_index = data_diff_mean.abs().idxmax()

        # Split data into inputs (before max change) and labels (after max change)
        inputs = torch.tensor(data.iloc[:max_index, :].values, dtype=torch.float32)
        labels = torch.tensor(data.iloc[max_index:, :].values, dtype=torch.float32)

        file_name = os.path.basename(file_path)

        return inputs, labels, file_name


def create_dataloader(data_dir, batch_size=32, shuffle=True, test_size=0.2, random_state=42):
    """
    Creates training and testing dataloaders from the data directory.

    Args:
        data_dir (str): The directory containing the data files.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the training data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator for splitting.

    Returns:
        tuple: A tuple containing the training DataLoader and testing DataLoader.
    """
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=random_state)

    train_dataset = BallDataset(data_dir, train_files)
    test_dataset = BallDataset(data_dir, test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, test_dataloader


class MultiHeadAttentionModel(nn.Module):
    """A sequence-to-sequence model using Multi-Head Attention."""

    def __init__(self, input_size, hidden_size, num_heads, output_size, num_layers=2):
        super(MultiHeadAttentionModel, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.fc(x)
        attn_output = x
        for attn_layer in self.attention_layers:
            attn_output, _ = attn_layer(attn_output, attn_output, attn_output)

        output = self.fc_out(attn_output)
        return output


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        """Adds positional encoding to the input tensor."""
        return x + self.encoding[:, :x.size(1)].to(x.device)


class LSTMModelWithPositionEncoding(nn.Module):
    """An LSTM-based model with Positional Encoding."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(LSTMModelWithPositionEncoding, self).__init__()
        self.positional_encoding = PositionalEncoding(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.positional_encoding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc_out(lstm_out)
        return output


def train_and_save_model(model, train_loader, criterion, optimizer, device, seq_len=1, num_epochs=20, save_path="model_weights.pth"):
    """
    Trains the model and saves its weights.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
        seq_len (int): The length of the sequence to predict. If 0, predicts the full target sequence length.
        num_epochs (int): The number of training epochs.
        save_path (str): The path to save the model weights.
    """
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, input_lengths, targets, target_lengths, file_names in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            current_input = inputs.clone()
            loss = 0
            # Dynamically determine max_length based on target lengths if seq_len is 0
            max_length = target_lengths.max().item() if seq_len == 0 else seq_len

            all_outputs = []

            # Autoregressive prediction loop
            for t in range(max_length):
                outputs = model(current_input)
                next_preds = []
                for i, length in enumerate(target_lengths):
                    if t < length:
                        # Get the prediction for the current time step
                        next_pred = outputs[i, t, :]
                        next_preds.append(next_pred)
                    else:
                        # Pad if the sequence is shorter than the current time step
                        next_preds.append(torch.zeros_like(outputs[i, t, :]))
                next_preds = torch.stack(next_preds, dim=0)
                next_preds = next_preds.unsqueeze(1)
                # Append the new prediction to the input for the next time step
                current_input = torch.cat((current_input, next_preds), dim=1)
                all_outputs.append(next_preds)

            all_outputs = torch.cat(all_outputs, dim=1)

            # Calculate loss only on the valid parts of the sequences
            mask = torch.arange(max_length, device=device).expand(len(target_lengths), max_length) < target_lengths.unsqueeze(1).to(device)
            mask = mask.unsqueeze(-1).expand(-1, -1, targets.size(-1))  # Adjust mask shape to match targets
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

    # Save the model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def test_model(model, test_loader, criterion, device, seq_len=1, load_path=None):
    """
    Tests the model on the test dataset.

    Args:
        model (nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for the test data.
        criterion: The loss function.
        device (torch.device): The device to run the test on.
        seq_len (int): The length of the sequence to predict. If 0, predicts the full target sequence length.
        load_path (str, optional): Path to the saved model weights. If provided, loads the weights.

    Returns:
        float: The average test loss.
    """
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

            # Autoregressive prediction loop
            for t in range(max_length):
                outputs = model(current_input)
                next_preds = []
                for i, length in enumerate(target_lengths):
                    if t < length:
                        next_pred = outputs[i, t, :]
                        next_preds.append(next_pred)
                    else:
                        # Pad if the sequence is shorter
                        next_preds.append(torch.zeros_like(outputs[i, t, :]))
                next_preds = torch.stack(next_preds, dim=0)
                next_preds = next_preds.unsqueeze(1)
                current_input = torch.cat((current_input, next_preds), dim=1)
                all_outputs.append(next_preds)

            all_outputs = torch.cat(all_outputs, dim=1)

            # Create mask to calculate loss only on valid parts of the sequence
            mask = torch.arange(max_length, device=device).expand(len(target_lengths), max_length) < target_lengths.unsqueeze(1).to(device)
            mask = mask.unsqueeze(-1).expand(-1, -1, targets.size(-1))
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
    # Set seed for reproducibility
    set_seed(42)

    # --- Configuration ---
    num_epochs = 100
    data_dir = 'data/data_collection'
    save_path = 'ckpt/model_weights.pth'
    batch_size = 12
    input_size = 6   # Dimension of input features
    hidden_size = 64 # Hidden size for attention layers
    output_size = 6  # Dimension of output features
    num_heads = 6    # Number of heads in Multi-Head Attention

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    train_dataloader, test_dataloader = create_dataloader(data_dir, batch_size=batch_size)

    # --- Model, Loss, and Optimizer ---
    model = LSTMModelWithPositionEncoding(input_size, hidden_size, output_size, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training and Testing ---
    # seq_len=0 means the model will predict the full length of the target sequence
    train_and_save_model(model, train_dataloader, criterion, optimizer, device, seq_len=0, num_epochs=num_epochs, save_path=save_path)
    test_model(model, test_dataloader, criterion, device, seq_len=0, load_path=save_path)
