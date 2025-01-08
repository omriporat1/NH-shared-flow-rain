import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# Load loss data from files
def load_loss_data(files):
    all_losses = []
    for file in files:
        with open(file, 'r') as f:
            losses = [float(line.strip()) for line in f if line.strip() and not line.strip().lower() == 'nan']
            all_losses.extend(losses)
    return np.array(all_losses)

epoch_n = 1

# Define predefined axis ranges
HIST_X_RANGE = [1e-3, 1e0]  # X-axis range for histogram
HIST_Y_RANGE = [0, 0.17]    # Y-axis range for normalized distribution
CDF_X_RANGE = [1e-4, 1e1]   # X-axis range for CDF

# Load the data
validation_files = glob.glob("*validation*epoch_" + str(epoch_n) + ".*")
training_file = glob.glob("*training*epoch_" + str(epoch_n) + ".*")

training_losses_new = load_loss_data(training_file)
validation_losses_new = load_loss_data(validation_files)

# Define logarithmic bins for the histogram
log_bins_new = np.logspace(np.log10(HIST_X_RANGE[0]), np.log10(HIST_X_RANGE[1]), 30)

# Compute histograms with logarithmic bins
train_hist_log_new, _ = np.histogram(training_losses_new, bins=log_bins_new, density=False)
val_hist_log_new, _ = np.histogram(validation_losses_new, bins=log_bins_new, density=False)

# Compute mean, median, and standard deviation of the training and validation losses
train_mean_new = np.mean(training_losses_new)
train_median_new = np.median(training_losses_new)
train_std_new = np.std(training_losses_new)
valid_std_new = np.std(validation_losses_new)
valid_mean_new = np.mean(validation_losses_new)
valid_median_new = np.median(validation_losses_new)

# Normalize to get proportions
train_hist_log_new = train_hist_log_new / train_hist_log_new.sum()
val_hist_log_new = val_hist_log_new / val_hist_log_new.sum()

# Plot the histogram with logarithmic bins
bin_centers_log_new = 0.5 * (log_bins_new[:-1] + log_bins_new[1:])
width_log_new = (log_bins_new[1:] - log_bins_new[:-1]) * 0.4  # Adjusted bar width for clarity

plt.figure(figsize=(10, 6))
# plot training and validation histograms, add labels and legend with statistics:
plt.axvline(train_mean_new, color='blue', linestyle='--', label=f'Training Mean: {train_mean_new:.4f}, N={len(training_losses_new)}')
plt.axvline(train_median_new, color='blue', linestyle='-', label=f'Training Median: {train_median_new:.4f}, N={len(training_losses_new)}')
plt.axvline(valid_mean_new, color='orange', linestyle='--', label=f'Validation Mean: {valid_mean_new:.4f}, N={len(validation_losses_new)}')
plt.axvline(valid_median_new, color='orange', linestyle='-', label=f'Validation Median: {valid_median_new:.4f}, N={len(validation_losses_new)}')

plt.bar(bin_centers_log_new - width_log_new / 2, train_hist_log_new, width=width_log_new,
        label='Training', alpha=0.7, align='center')
plt.bar(bin_centers_log_new + width_log_new / 2, val_hist_log_new, width=width_log_new,
        label='Validation', alpha=0.7, align='center')

plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlim(HIST_X_RANGE)  # Set predefined X-axis range
plt.ylim(HIST_Y_RANGE)  # Set predefined Y-axis range
plt.title('Normalized Loss Distribution with Logarithmic Bins - Epoch' + str(epoch_n))
plt.xlabel('Loss (Log Scale)')
plt.ylabel('Proportion')
plt.legend()
plt.tight_layout()

# Save the histogram with logarithmic bins
plt.savefig('loss_distribution_log_bins_histogram_e' + str(epoch_n) + '.png')
plt.show()

# Compute the cumulative distribution functions (CDFs) for training and validation
train_sorted_new = np.sort(training_losses_new)
val_sorted_new = np.sort(validation_losses_new)

train_cdf_new = np.arange(1, len(train_sorted_new) + 1) / len(train_sorted_new)
val_cdf_new = np.arange(1, len(val_sorted_new) + 1) / len(val_sorted_new)

# Plot the CDFs
plt.figure(figsize=(10, 6))
plt.plot(train_sorted_new, train_cdf_new, label='Training CDF', linewidth=2)
plt.plot(val_sorted_new, val_cdf_new, label='Validation CDF', linewidth=2)

plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlim(CDF_X_RANGE)  # Set predefined X-axis range
plt.title('Cumulative Distribution Function (CDF) of Loss - Epoch ' + str(epoch_n))
plt.xlabel('Loss (Log Scale)')
plt.ylabel('Cumulative Proportion')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the CDF plot
plt.savefig('loss_cdf_curve_e' + str(epoch_n) + '.png')
plt.show()
