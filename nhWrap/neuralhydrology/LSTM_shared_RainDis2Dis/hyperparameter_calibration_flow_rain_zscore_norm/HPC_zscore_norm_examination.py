import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd


'''
def parse_hyperparameters(folder_name):
    """Extract hyperparameters from the folder name."""
    match = re.match(r".*hidden_size(\d+)_batch_size(\d+)_learning_rate(\d+).*", folder_name)
    if match:
        hidden_size = int(match.group(1))
        batch_size = int(match.group(2))
        learning_rate = 10*float(f"0.{match.group(3)}")
        return hidden_size, batch_size, learning_rate
    return None, None, None
'''


def parse_hyperparameters(folder_name):
    """
    Extract hyperparameters dynamically from the folder name.

    Parameters:
        folder_name (str): Name of the folder containing the run.

    Returns:
        dict: A dictionary of hyperparameters found in the folder name.
    """
    # Regular expressions for different parameters
    patterns = {
        "hidden_size": r"hidden_size(\d+)",
        "seq_length": r"seq_length(\d+)",
        "batch_size": r"batch_size(\d+)",
        "learning_rate": r"learning_rate([\d\.]+)",
    }

    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        if match:
            params[key] = int(match.group(1)) if "size" in key or "length" in key else float(match.group(1))

    return params


def plot_curves(curves, x_param, y_param, fixed_param, fixed_values, param_labels, title): # NEEDS ADJUSTMENT TO THE CHANGE IN PARSED HYPERPARAMETERS
    """Plot the loss and NSE curves for the 3x3 grid."""

    unique_x = {d[x_param] for d in curves}
    unique_y = {d[y_param] for d in curves}
    unique_fixed = {d[fixed_param] for d in curves}



    fig, axes = plt.subplots(len(unique_x), len(unique_y), figsize=(18, 18), sharex=True, sharey=True)
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
    # save the figure with file name as title:
    fig.savefig(title + ".png")


def save_curves_as_table_append(log_dir, output_file, filter_text="*"):
    """
    Extract loss and NSE curves from TensorBoard logs and append to an existing CSV table.

    Parameters:
        log_dir (str): Path to the directory containing TensorBoard log folders.
        output_file (str): Path to the output CSV file.
        filter_text (str): Text to filter folders (e.g., "minmax"). Use "*" to include all folders.
    """
    # Check if the file exists; if it does, load it into a DataFrame
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
    else:
        existing_df = pd.DataFrame()

    data = []
    for folder in os.listdir(log_dir):
        run_path = os.path.join(log_dir, folder)
        if not os.path.isdir(run_path):
            continue

        # Apply filter if specified
        if filter_text != "*" and filter_text not in folder:
            continue

        params = parse_hyperparameters(folder)
        if not params:
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

        # Ensure all lists have the same length for proper DataFrame creation
        max_len = max(len(train_loss), len(valid_loss), len(valid_nse))
        train_loss.extend([None] * (max_len - len(train_loss)))
        valid_loss.extend([None] * (max_len - len(valid_loss)))
        valid_nse.extend([None] * (max_len - len(valid_nse)))
        steps.extend(range(len(steps), max_len))

        for i, step in enumerate(steps):
            row = {
                'file_name': folder,
                'step': step,
                'train_loss': train_loss[i],
                'valid_loss': valid_loss[i],
                'valid_nse': valid_nse[i],
            }
            row.update(params)  # Dynamically add extracted parameters
            data.append(row)

    # Convert new data to pandas DataFrame
    new_df = pd.DataFrame(data)

    # Combine existing and new data, avoiding duplicates
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates()
    else:
        combined_df = new_df

    # Save the updated DataFrame back to the CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Data updated and saved to {output_file}")


def extract_curves(log_dir): # NEEDS ADJUSMENT TO THE CHANGE IN PARSED HYPERPARAMETERS
    """Extract loss and NSE curves from TensorBoard logs."""
    curves = []
    for folder in os.listdir(log_dir):
        # run_path = os.path.join(log_dir, folder)
        run_path = log_dir
        if not os.path.isdir(run_path):
            continue

        # hidden_size, batch_size, learning_rate = parse_hyperparameters(folder)
        hidden_size, batch_size, learning_rate = parse_hyperparameters(run_path)
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
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            # 'valid_nse': valid_nse,
            'steps': steps
        })

    return curves


def extract_all_curves(log_dir): # NEEDS ADJUSTMENT TO THE CHANGE IN PARSED HYPERPARAMETERS
    """Extract all logged scalar parameters from TensorBoard logs."""
    curves = []

    for folder in os.listdir(log_dir):
        # run_path = os.path.join(log_dir, folder)
        run_path = log_dir
        if not os.path.isdir(run_path):
            continue

        # hidden_size, batch_size, learning_rate = parse_hyperparameters(folder)
        hidden_size, batch_size, learning_rate = parse_hyperparameters(run_path)

        if hidden_size is None:
            continue

        event_acc = EventAccumulator(run_path)
        event_acc.Reload()

        # Get all scalar tags
        scalar_tags = event_acc.Tags().get('scalars', [])

        # Initialize a dictionary to hold scalar values for this run
        run_data = {
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'steps': []
        }

        # Extract scalar data dynamically
        scalar_data = {tag: [] for tag in scalar_tags}
        for tag in scalar_tags:
            for event in event_acc.Scalars(tag):
                scalar_data[tag].append(event.value)
                if tag == 'train/avg_loss':  # Assume steps are tied to training loss
                    run_data['steps'].append(event.step)

        # Add extracted scalar data to run_data
        run_data.update(scalar_data)

        curves.append(run_data)

    return curves

def main():
    log_dir = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\LSTM_shared_RainDis2Dis\runs")
    #            r"\Lag18_Basins8_2024_12_16")  # Replace with your TensorBoard logs folder path
    # log_dir = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\LSTM_shared_RainDis2Dis\runs"
    #            r"\Check_loss_zscore_norm_ensemble_hidden_size128_batch_size512_learning_rate0001_3012_133227")  # Replace with your TensorBoard logs folder path
    '''
    curves = extract_curves(log_dir)
    all_curves = extract_all_curves(log_dir)
    # Generate the 3x3 tables
    fixed_params = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [512, 2048],
        'hidden_size': [64, 128, 256]
    }

    param_labels = {
        'learning_rate': 'Learning Rate',
        'batch_size': 'Batch Size',
        'hidden_size': 'Hidden Size'
    }

    plot_curves(
        curves, 'learning_rate', 'batch_size', 'hidden_size', fixed_params['hidden_size'], param_labels,
        title="Effect of Learning Rate and Batch Size (Fixed Hidden Size)"
    )

    plot_curves(
        curves, 'learning_rate', 'hidden_size', 'batch_size', fixed_params['batch_size'], param_labels,
        title="Effect of Learning Rate and Hidden Size (Fixed Batch Size)"
    )

    plot_curves(
        curves, 'batch_size', 'hidden_size', 'learning_rate', fixed_params['learning_rate'], param_labels,
        title="Effect of Batch Size and Hidden Size (Fixed Learning Rate)"
    )
    '''

    save_curves_as_table_append(log_dir, "loss_data_for_report.csv", filter_text="hidden_size64_seq_length36_learning_rate0001_flow_only_training_zscore_norm_2512_162024")
# Main execution


if __name__ == '__main__':
    main()
