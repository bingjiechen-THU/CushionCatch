import torch
import numpy as np
from Compliance_Learner.main_mhsa_lstm import LSTMModelWithPositionEncoding, set_seed


class POC_Traj_Learner():
    """
    A class for loading a pre-trained trajectory prediction model and performing inference.
    """

    def __init__(self, load_path="./ckpt/model_weights.pth"):
        """
        Initializes the trajectory learner.

        Args:
            load_path (str): Path to the pre-trained model weights.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 6  # Dimension of input features
        hidden_size = 64  # Hidden size of the LSTM layers
        output_size = 6  # Dimension of output features
        self.model = LSTMModelWithPositionEncoding(input_size, hidden_size, output_size, num_layers=2).to(device)

        # Load model weights
        self.load_model_weights(load_path, device)

    def load_model_weights(self, load_path, device):
        """
        Loads model weights from a specified path.

        Args:
            load_path (str): The path to the model weights file.
            device (torch.device): The device to load the model onto.
        """
        self.model.load_state_dict(torch.load(load_path))
        self.model.to(device)
        self.model.eval()  # Switch to evaluation mode
        print(f"Model weights loaded from {load_path}")

    def inference(self, inputs, device, max_length=100):
        """
        Performs autoregressive inference to predict a sequence.

        Args:
            inputs (np.ndarray): The initial input sequence.
            device (torch.device): The device to perform computation on.
            max_length (int): The maximum length of the sequence to generate.

        Returns:
            np.ndarray: The generated sequence of predictions.
        """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)

            # Initialize current input with the original input
            current_input = inputs.clone()

            # Autoregressively generate the sequence up to max_length
            for t in range(max_length):
                # Generate the prediction for the next time step
                outputs = self.model(current_input)
                next_pred = outputs[:, -1, :]  # Get the prediction from the last time step

                # Append the prediction to the input sequence for the next step
                next_pred = next_pred.unsqueeze(1)
                current_input = torch.cat((current_input, next_pred), dim=1)

                # Save the prediction
                predictions.append(next_pred.cpu().numpy())

        print("Inference complete. Predictions generated.")
        predictions = np.concatenate(predictions, axis=1)
        return predictions

    def predict_vel(self, vel_posi_input):
        """
        Predicts a trajectory of future velocities based on an initial state.

        Args:
            vel_posi_input (list or np.ndarray): The initial state containing
                                                 velocity and position (6 dimensions).

        Returns:
            np.ndarray: An array of predicted velocities (N, 3) where N is max_length.
        """
        set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_length = 16  # Maximum length of the sequence to generate
        input_size = 6  # Dimension of input features

        # Input sequence must have a batch dimension. The output will also have a batch dimension.
        inputs = np.array(vel_posi_input).reshape(1, 1, input_size)

        # Predict a trajectory of max_length steps (xyz velocity and position)
        predictions = self.inference(inputs, device, max_length=max_length)

        # Remove the batch dimension
        predictions = predictions[0]

        # Return only the velocity predictions
        return predictions[:, :3]
