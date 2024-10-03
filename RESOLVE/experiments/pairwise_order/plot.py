import numpy as np
import matplotlib.pyplot as plt

# Load the saved accuracies
accuracies_mlp = np.load('final_accuracies_mlp.npy')
accuracies_len = np.load('final_accuracies_len.npy')
accuracies_transformer = np.load('final_accuracies_transformer.npy')
accuracies_rel_nvsa_train = np.load('final_accuracies_resolve.npy')
accuracies_abstractor = np.load('final_accuracies_abstractor.npy')
accuracies_predinet = np.load('final_accuracies_predinet.npy')
accuracies_corelnet = np.load('final_accuracies_correlnet.npy')




# Define model names for labeling and plotting
models = [ 
    "Transformer","RESOLVE", 
    "Abstractor", "PrediNet", "CorelNet", "MLP", "LEN"
]

# Combine the arrays for easy access
all_accuracies = [
    accuracies_transformer,accuracies_rel_nvsa_train, 
    accuracies_abstractor, accuracies_predinet, accuracies_corelnet, accuracies_mlp, accuracies_len
]



print(accuracies_rel_nvsa_train.mean(axis=0)[[0,2,3,6,10,18,20]])
print(accuracies_transformer.mean(axis=0)[[0,2,3,6,10,18,20]])
print(accuracies_abstractor.mean(axis=0)[[0,2,3,6,10,18,20]])
color_prime = ['red', 'green', 'orange', 'blue', 'navy', 'gray', 'purple', 'black']

indices = [1,4,6,7,10,12,17]
# Define training sizes (the x-axis) and exclude the last one
training_sizes = np.array(range(10, 250, 10))[[0,2,3,6,10,18,20]].tolist()


# Calculate the mean accuracy across seeds for each model and training size, excluding the last training size
mean_accuracies = [np.mean(acc, axis=0)[[0,2,3,6,10,18,20]] for acc in all_accuracies]  # Remove the last element for each model

# Define the colors and alpha values
same_color = 'purple'
alpha_values = [0.3, 0.5, 0.7, 0.9]  # Different alpha values for NVSA models
colors = {}

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))
bar_width = 0.08  # Width of the bars
index = np.arange(len(training_sizes))  # X locations for the groups
j = 0

# Plot each model's accuracies as a group of bars with slight offsets
for i, (model_name, model_accuracies) in enumerate(zip(models, mean_accuracies)):
    if model_name in colors:
        ax.bar(index + i * bar_width, model_accuracies, bar_width, 
               label=model_name, color=colors[model_name][0], alpha=colors[model_name][1])
    else:
        ax.bar(index + i * bar_width, model_accuracies, bar_width, label=model_name, color=color_prime[j])
        j += 1

# Set labels and title with increased font sizes
ax.set_xlabel('Training Size', fontsize=28, fontweight='bold')
ax.set_ylabel('Mean Accuracy', fontsize=28, fontweight='bold')
#ax.set_title('Mean Accuracy Across Models and Training Sizes', fontsize=16)

# Set x-axis tick labels with increased font size
ax.set_xticks(index + bar_width * (len(models) / 2))  # Center the xticks
ax.set_xticklabels(training_sizes, fontsize=35)

# Customize the font size of y-axis ticks
ax.tick_params(axis='y', labelsize=35)

# Rotate x-axis tick labels
plt.xticks(rotation=45)

# Increase the legend font size
#ax.legend(fontsize=12)

# Add space between groups
plt.tight_layout()

# Show the plot
plt.show()

