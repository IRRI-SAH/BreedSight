import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from BreedSightTuning import run_cross_validation

# Define file paths
training_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/training_phenotypic_data.csv"
training_additive_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/training_additive.csv"
testing_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/testing_data.csv"
testing_additive_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/testing_additive.csv"

# Run cross-validation
results_dict = run_cross_validation(
    training_file=training_file_path,
    training_additive_file=training_additive_file_path,
    testing_file=testing_file_path,
    testing_additive_file=testing_additive_file_path,
    feature_selection=True,
    #learning_rate=0.001
)

# Print results safely
print("Training predictions:")
print(results_dict['train_predictions'].head() if results_dict['train_predictions'] is not None else "No training predictions available")

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
save_data(results_dict['train_predictions'], "train_predictions")
save_data(results_dict['val_predictions'], "val_predictions")
save_data(results_dict['test_predictions'], "test_predictions")
save_data(results_dict['feature_importances'], "feature_importances")
save_data(results_dict['results'], "results_summary")

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
handle_plot(results_dict['train_plot'], "train_plot")
handle_plot(results_dict['val_plot'], "val_plot")
handle_plot(results_dict['test_plot'], "test_plot")

# Save additional CSV files
save_data(results_dict['train_csv'], "train_additional")
save_data(results_dict['val_csv'], "val_additional")
save_data(results_dict['test_csv'], "test_additional")

print("\nProcessing complete. Results saved in 'output' directory.")
print("Contents of output directory:")
print(os.listdir("output"))