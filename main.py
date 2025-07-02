import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from BreedSight import run_cross_validation # change BreedSight to BreedSightTuning if you're using BreedSightTuning Model#

# Define file paths
training_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/training_phenotypic_data.csv"
training_additive_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/training_additive.csv"
testing_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/testing_data.csv"
testing_additive_file_path = "C:/Users/Ashmitha/Desktop/BreedSight/BreedSight/Example_files/testing_additive.csv"

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
print(train_pred.head() if train_pred is not None else "No training predictions available")

# Create output directory
os.makedirs("output", exist_ok=True)

# Save predictions
def save_data(data, filename):
    """Save data to CSV file, handling both DataFrames and file paths"""
    if data is None:
        print(f"No {filename} data available")
        return
    
    output_path = f"output/{filename}.csv"
    
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
            print(f"Saved DataFrame to {output_path}")
        elif isinstance(data, str):  # If it's a file path
            if os.path.exists(data):
                shutil.copy(data, output_path)
                print(f"Copied {data} to {output_path}")
            else:
                print(f"File {data} does not exist")
        else:
            print(f"Unsupported data type for {filename}: {type(data)}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Save prediction files
save_data(train_pred, "train_predictions")
save_data(val_pred, "val_predictions")
save_data(test_pred, "test_predictions")

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

# Save additional CSV files
save_data(train_csv, "train_additional")
save_data(val_csv, "val_additional")
save_data(test_csv, "test_additional")

print("\nProcessing complete. Results saved in 'output' directory.")
print("Contents of output directory:")
print(os.listdir("output"))
