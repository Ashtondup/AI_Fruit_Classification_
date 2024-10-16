import os
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = r"C:/Users/ashto/OneDrive/Desktop/Hons/AI_Fruit_Classification_/Metrics/ROC-AUC.txt"

# Function to read the file and extract FPR, TPR, and AUC values
def read_roc_auc_file(file_path):
    tprs = []
    fprs = []
    aucs = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Try to extract the values, skip non-numeric lines
            try:
                # Assuming the file has FPR, TPR, and AUC separated by space or comma
                values = line.strip().split()  # Adjust this if using commas
                fpr, tpr, auc = map(float, values)
                fprs.append(fpr)
                tprs.append(tpr)
                aucs.append(auc)
            except ValueError:
                # Skip lines that cannot be parsed to float (e.g., headers)
                print(f"Skipping line: {line.strip()}")
                continue
    
    return np.array(tprs), np.array(fprs), np.array(aucs)

# Function to calculate averages and plot the ROC curve
def plot_average_roc_auc(tprs, fprs, aucs):
    # Calculate the average TPR, FPR, and AUC per epoch
    avg_tpr = np.mean(tprs)
    avg_fpr = np.mean(fprs)
    avg_auc = np.mean(aucs)
    
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, 'b', alpha=0.5, label="ROC per epoch")
    plt.plot(avg_fpr, avg_tpr, 'ro', label=f"Avg ROC (FPR: {avg_fpr:.2f}, TPR: {avg_tpr:.2f})")
    
    # Adding labels and title
    plt.title(f"ROC Curve with Average TPR/FPR (Avg AUC: {avg_auc:.2f})")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Main execution
if __name__ == "__main__":
    if os.path.exists(file_path):
        tprs, fprs, aucs = read_roc_auc_file(file_path)
        plot_average_roc_auc(tprs, fprs, aucs)
    else:
        print(f"File not found at {file_path}")
