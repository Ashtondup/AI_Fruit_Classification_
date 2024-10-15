import matplotlib.pyplot as plt

def extract_float_from_line(line):
    """Extracts the float value from a line in the format '<Metric> = <float>'."""
    try:
        return float(line.split('=')[1].strip())
    except (IndexError, ValueError) as e:
        print(f"Error converting value: {line} -> {e}")
        return None

def read_metrics(file_path):
    """Reads metrics from a file and returns a list of floats."""
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            value = extract_float_from_line(line)
            if value is not None:
                values.append(value)
    return values

def plot_metric(metric_values, metric_name, color):
    """Plots a single metric on its own graph."""
    epochs = list(range(1, len(metric_values) + 1))  # X-axis: Epochs
    plt.figure()  # Create a new figure for each metric
    plt.plot(epochs, metric_values, marker='o', label=metric_name, color=color)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all_metrics(accuracy_file, loss_file, f1_score_file):
    # Read the metrics from files
    accuracies = read_metrics(accuracy_file)
    losses = read_metrics(loss_file)
    f1_scores = read_metrics(f1_score_file)

    # Plot each metric separately
    if accuracies:
        plot_metric(accuracies, 'Accuracy', 'blue')
    if losses:
        plot_metric(losses, 'Loss', 'red')
    if f1_scores:
        plot_metric(f1_scores, 'F1 Score', 'green')

# File paths to your metrics files
accuracy_file = r"C:/Users/ashto/OneDrive/Desktop/Hons/AI_Fruit_Classification_/Metrics/Accuracy.txt"
loss_file = r"C:/Users/ashto/OneDrive/Desktop/Hons/AI_Fruit_Classification_/Metrics/Loss.txt"
f1_score_file = r"C:/Users/ashto/OneDrive/Desktop/Hons/AI_Fruit_Classification_/Metrics/F1_score.txt"

# Plot all metrics separately
plot_all_metrics(accuracy_file, loss_file, f1_score_file)
