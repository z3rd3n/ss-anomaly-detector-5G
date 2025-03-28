import matplotlib.pyplot as plt

# Define the thresholds and the associated metrics
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

f1_scores = [0.3208, 0.3926, 0.4486, 0.4937, 0.5316, 0.5667, 0.5957,
             0.6223, 0.6440, 0.6653, 0.6825, 0.6987, 0.7164, 0.7331,
             0.7472, 0.7644, 0.7805, 0.7950]  # Optimal threshold at 0.95

# Detection rates per class for each threshold
# Order: [Class 0 (Normal), Class 1 (Type 1), Class 2 (Type 2), Class 3 (Type 3), Class 4 (Type 4)]
det_rates_class0 = [0.9582, 0.9702, 0.9770, 0.9814, 0.9846, 0.9870, 0.9888,
                    0.9903, 0.9915, 0.9925, 0.9934, 0.9941, 0.9949, 0.9956,
                    0.9962, 0.9969, 0.9975, 0.9978]
det_rates_class1 = [0.9658, 0.9566, 0.9450, 0.9362, 0.9285, 0.9210, 0.9159,
                    0.9091, 0.9036, 0.8988, 0.8917, 0.8844, 0.8778, 0.8710,
                    0.8628, 0.8530, 0.8430, 0.8330]
det_rates_class2 = [0.6106, 0.5413, 0.4785, 0.3927, 0.2871, 0.2475, 0.1980,
                    0.1617, 0.1320, 0.1155, 0.0957, 0.0660, 0.0561, 0.0462,
                    0.0264, 0.0231, 0.0132, 0.0033]
det_rates_class3 = [0.6584, 0.5693, 0.5050, 0.4257, 0.3762, 0.3465, 0.3020,
                    0.2822, 0.2376, 0.2079, 0.1931, 0.1337, 0.1188, 0.1040,
                    0.0792, 0.0396, 0.0099, 0.0000]
det_rates_class4 = [1.0000] * len(thresholds)  # Always perfect detection

# Create a figure with two subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Plot overall F1 Score vs Threshold
ax[0].plot(thresholds, f1_scores, marker='o', linestyle='-', color='blue', label='F1 Score')
ax[0].axvline(x=0.95, color='red', linestyle='--', label='Optimal Threshold (0.95)')
ax[0].set_title('F1 Score vs Threshold')
ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('F1 Score')
ax[0].set_ylim(0.3, 0.85)
ax[0].grid(True)
ax[0].legend()

# Plot per-class detection rates vs Threshold
ax[1].plot(thresholds, det_rates_class0, marker='o', linestyle='-', label='Class 0 (Normal)')
ax[1].plot(thresholds, det_rates_class1, marker='s', linestyle='-', label='Class 1 (Type 1)')
ax[1].plot(thresholds, det_rates_class2, marker='^', linestyle='-', label='Class 2 (Type 2)')
ax[1].plot(thresholds, det_rates_class3, marker='d', linestyle='-', label='Class 3 (Type 3)')
ax[1].plot(thresholds, det_rates_class4, marker='v', linestyle='-', label='Class 4 (Type 4)')
ax[1].set_title('Detection Rates vs Threshold')
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Detection Rate')
ax[1].set_ylim(0.0, 1.05)
ax[1].grid(True)
ax[1].legend()

# Add a main title and adjust layout
plt.suptitle('Ablation Study: Effect of Threshold Selection', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure for publication (optional)
plt.savefig("threshold_ablation_study.png", dpi=300)

# Show the plot
plt.show()
