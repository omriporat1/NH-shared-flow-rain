import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import re
import numpy as np


def parse_hyperparameters(folder_name):
    """Extract hyperparameters from the folder name."""
    match = re.match(r".*hidden_size(\d+)_seq_length(\d+)_learning_rate(\d+).*", folder_name)
    if match:
        hidden_size = int(match.group(1))
        seq_length = int(match.group(2))
        learning_rate = 10*float(f"0.{match.group(3)}")
        return hidden_size, seq_length, learning_rate
    return None, None, None


def plot_curves(curves, x_param, y_param, fixed_param, fixed_values, param_labels, title):
    """Plot the loss and NSE curves for the 3x3 grid."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16)

    # Determine uniform Y-axis ranges
    all_train_losses = [val for curve in curves for val in curve['train_loss']]
    all_valid_losses = [val for curve in curves for val in curve['valid_loss']]
    # all_nses = [val for curve in curves for val in curve['valid_nse']]
    # nse_y_range = (min(all_nses), max(all_nses))

    loss_y_range = (min(all_train_losses + all_valid_losses), max(all_train_losses + all_valid_losses))

    x_values = sorted(set(curve[x_param] for curve in curves))
    y_values = sorted(set(curve[y_param] for curve in curves))

    color_map = {fixed: f"C{idx}" for idx, fixed in enumerate(fixed_values)}

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            ax = axes[i, j]
            ax.set_title(f"{param_labels[x_param]}={x}, {param_labels[y_param]}={y}")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_ylim(loss_y_range)

            '''
            ax2 = ax.twinx()
            ax2.set_ylabel('NSE', color='tab:blue')
            ax2.set_ylim(nse_y_range)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            '''

            sub_curves = [
                curve for curve in curves if curve[x_param] == x and curve[y_param] == y
            ]

            if not sub_curves:
                # Add a message to indicate missing data
                ax.text(0.5, 0.5, "No Data", fontsize=12, ha='center', va='center')
                continue

            for fixed in fixed_values:
                fixed_curves = [
                    curve for curve in sub_curves if curve[fixed_param] == fixed
                ]

                if not fixed_curves:
                    # Skip if no data for this fixed_param
                    continue

                for curve in fixed_curves:
                    steps = curve['steps']
                    train_loss = curve['train_loss']
                    valid_loss = curve['valid_loss']
                    # valid_nse = curve['valid_nse']

                    color = color_map[fixed]
                    label = f"{param_labels[fixed_param]}={fixed}"

                    # Plot train/validation loss
                    ax.plot(steps, train_loss, label=f"{label} (train loss)", linestyle='-', color=color)
                    ax.plot(steps, valid_loss, label=f"{label} (valid loss)", linestyle='--', color=color)

                    # Plot NSE on the secondary axis
                    # ax2.plot(steps, valid_nse, linestyle=':', color=color)

    # Create a single legend for all subplots
    legend_labels = [f"{param_labels[fixed_param]}={fixed}" for fixed in fixed_values]
    legend_handles = [plt.Line2D([0], [0], color=color_map[fixed], label=label) for fixed, label in
                      zip(fixed_values, legend_labels)]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(fixed_values), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def extract_curves(log_dir):
    """Extract loss and NSE curves from TensorBoard logs."""
    curves = []
    for folder in os.listdir(log_dir):
        run_path = os.path.join(log_dir, folder)
        if not os.path.isdir(run_path):
            continue

        hidden_size, seq_length, learning_rate = parse_hyperparameters(folder)
        if hidden_size is None:
            continue

        event_acc = EventAccumulator(run_path)
        event_acc.Reload()

        train_loss = []
        valid_loss = []
        valid_nse = []
        steps = []

        if 'train/avg_loss' in event_acc.Tags()['scalars']:
            for e in event_acc.Scalars('train/avg_loss'):
                train_loss.append(e.value)
                steps.append(e.step)

        if 'valid/avg_loss' in event_acc.Tags()['scalars']:
            for e in event_acc.Scalars('valid/avg_loss'):
                valid_loss.append(e.value)

        if 'valid/mean_nse' in event_acc.Tags()['scalars']:
            for e in event_acc.Scalars('valid/mean_nse'):
                valid_nse.append(e.value)

        curves.append({
            'hidden_size': hidden_size,
            'seq_length': seq_length,
            'learning_rate': learning_rate,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            # 'valid_nse': valid_nse,
            'steps': steps
        })

    return curves


def main():
    log_dir = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\LSTM_shared_RainDis2Dis\runs"
               r"\HPC_training_zscore_norm_2024_12_26")  # Replace with your TensorBoard logs folder path
    curves = extract_curves(log_dir)

    # Generate the 3x3 tables
    fixed_params = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'seq_length': [36, 72, 144],
        'hidden_size': [16, 32, 64]
    }

    param_labels = {
        'learning_rate': 'Learning Rate',
        'seq_length': 'Sequence Length',
        'hidden_size': 'Hidden Size'
    }

    plot_curves(
        curves, 'learning_rate', 'seq_length', 'hidden_size', fixed_params['hidden_size'], param_labels,
        title="Effect of Learning Rate and Sequence Length (Fixed Hidden Size)"
    )

    plot_curves(
        curves, 'learning_rate', 'hidden_size', 'seq_length', fixed_params['seq_length'], param_labels,
        title="Effect of Learning Rate and Hidden Size (Fixed Sequence Length)"
    )

    plot_curves(
        curves, 'seq_length', 'hidden_size', 'learning_rate', fixed_params['learning_rate'], param_labels,
        title="Effect of Sequence Length and Hidden Size (Fixed Learning Rate)"
    )
# Main execution


if __name__ == '__main__':
    main()
