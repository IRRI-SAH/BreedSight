import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from BreedSight import run_cross_validation

# Define file paths
training_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/phenotypic_data.csv"
training_additive_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/Training_additive.csv"
testing_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/Testing_data.csv"
testing_additive_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/Testing_addtive.csv"

# Run cross-validation
results = run_cross_validation(
    training_file=training_file_path,
    training_additive_file=training_additive_file_path,
    testing_file=testing_file_path,
    testing_additive_file=testing_additive_file_path,
    feature_selection=True,
    learning_rate=0.001
)

# Unpack the returned tuple
train_pred, val_pred, test_pred, train_plot, val_plot, test_plot, train_csv, val_csv, test_csv = results

# Print results safely
print("Training predictions:")
print(train_pred.head())

# Create output directory
os.makedirs("output", exist_ok=True)

# Save predictions
train_pred.to_csv("output/train_predictions.csv", index=False)
val_pred.to_csv("output/val_predictions.csv", index=False)
test_pred.to_csv("output/test_predictions.csv", index=False)

def handle_plot(plot, plot_name):
    """Handle saving plot whether it's a matplotlib object or file path."""
    if plot is None:
        print(f"No {plot_name} available")
        return
    
    output_path = f"output/{plot_name}.png"
    
    if hasattr(plot, 'figure'):  # Matplotlib figure object
        try:
            plot.figure.savefig(output_path)
            plt.close(plot.figure)
            print(f"Saved {plot_name} plot as {output_path}")
        except Exception as e:
            print(f"Error saving {plot_name} plot: {str(e)}")
    elif isinstance(plot, str):  # File path
        try:
            if os.path.exists(plot):
                shutil.copy(plot, output_path)
                print(f"Copied {plot_name} from {plot} to {output_path}")
            else:
                print(f"Plot file {plot} does not exist")
        except Exception as e:
            print(f"Error copying {plot_name} plot: {str(e)}")
    else:
        print(f"Unsupported plot type for {plot_name}: {type(plot)}")

# Handle plots
handle_plot(train_plot, "train_plot")
handle_plot(val_plot, "val_plot")
handle_plot(test_plot, "test_plot")

# Save additional CSV files if they exist
def save_csv(data, filename):
    if data is not None:
        try:
            data.to_csv(f"output/{filename}.csv", index=False)
            print(f"Saved {filename}.csv")
        except Exception as e:
            print(f"Error saving {filename}.csv: {str(e)}")
    else:
        print(f"No {filename} data available")

save_csv(train_csv, "train_additional")
save_csv(val_csv, "val_additional")
save_csv(test_csv, "test_additional")

print("Processing complete. Results saved in 'output' directory.")
    
