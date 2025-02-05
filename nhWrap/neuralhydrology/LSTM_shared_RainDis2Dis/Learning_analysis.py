import os
import yaml
import torch
import matplotlib.pyplot as plt

# Directory containing training files
directory = "runs/LSTM_shared_debugging_BadDataNans_18lags_0912_185508"

# Parse train_data_scaler.yml
scaler_file = os.path.join(directory, "train_data/train_data_scaler.yml")
with open(scaler_file, 'r') as f:
    scalers = yaml.safe_load(f)

# Initialize lists for epochs, losses, and metrics
epochs = []
losses = []
learning_rates = []

# Iterate over epoch files
for file in os.listdir(directory):
    if file.startswith("model_epoch") and file.endswith(".pt"):
        epoch = int(file.split("epoch")[1].split(".")[0])
        epochs.append(epoch)

        # Load model state
        model_path = os.path.join(directory, file)
        model_state = torch.load(model_path)

        # Extract loss (assuming it's stored as a scalar in state_dict or similar)
        if "loss" in model_state:
            losses.append(model_state["loss"])
        else:
            losses.append(None)  # Placeholder if not found

        # Load optimizer state for learning rate (optional)
        optimizer_path = os.path.join(directory, f"optimizer_state_epoch{str(epoch).zfill(3)}.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path)
            lr = optimizer_state['param_groups'][0]['lr']  # Assumes standard PyTorch optimizer
            learning_rates.append(lr)

# Sort data by epoch
sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
epochs = [epochs[i] for i in sorted_indices]
losses = [losses[i] for i in sorted_indices]
learning_rates = [learning_rates[i] for i in sorted_indices] if learning_rates else []

# Plot learning curves
plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, marker='o', label='Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()

# Plot Learning Rate if available
if learning_rates:
    plt.subplot(1, 2, 2)
    plt.plot(epochs, learning_rates, marker='o', label='Learning Rate', color='orange')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()

#%%

from tensorboard.backend.event_processing import event_accumulator

# Path to your TensorBoard log file
log_file_path = r'C:/TensorboardLogs/SharedLSTM/events.out.tfevents.1734013202.es-hydrolab13.14220'

# Load the TensorBoard log file
event_acc = event_accumulator.EventAccumulator(log_file_path)
event_acc.Reload()

# Print available tags
print("Available tags:", event_acc.Tags())

# Extract and analyze scalar data
scalars = event_acc.Scalars('your_tag_here')  # Replace with your desired tag
for scalar in scalars:
    print(f"Step: {scalar.step}, Value: {scalar.value}, Wall time: {scalar.wall_time}")
