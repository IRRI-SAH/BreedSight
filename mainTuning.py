import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from BreedSightTuning import run_cross_validation

def main():
    # Define file paths
    training_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/training_phenotypic_data.csv"
    training_additive_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/training_additive.csv"
    testing_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/testing_data.csv"
    testing_additive_file_path = "C:/Users/Dr Niranjani/Desktop/BreedSight-main/Example_files/testing_additive.csv"

    try:
        # Run cross-validation
        results = run_cross_validation(
            training_file=training_file_path,
            training_additive_file=training_additive_file_path,
            testing_file=testing_file_path,
            testing_additive_file=testing_additive_file_path,
            feature_selection=True,
            #learning_rate=0.001
        )

        # First inspect what we got back
        print(f"\nNumber of elements returned: {len(results)}")
        print("Types of returned elements:")
        for i, item in enumerate(results):
            print(f"Element {i}: {type(item)}")

        # Create output directory
        os.makedirs("output", exist_ok=True)

        # Save predictions function
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

        # Handle plots function
        def handle_plot(plot, plot_name):
            """Handle saving plot whether it's a matplotlib object or file path."""
            if plot is None:
                print(f"No {plot_name} available")
                return
            
            output_path = f"output/{plot_name}.png"
            
            if hasattr(plot, 'savefig'):  # Matplotlib figure object
                try:
                    plot.savefig(output_path)
                    plt.close(plot)
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

        

        # Save prediction files
        if len(results) > 0:
            save_data(results[0], "train_predictions")
            print("\nTraining predictions preview:")
            print(results[0].head() if isinstance(results[0], pd.DataFrame) else results[0])
        
        if len(results) > 1:
            save_data(results[1], "val_predictions")
        
        if len(results) > 2:
            save_data(results[2], "test_predictions")
        
       # if len(results) > 3:
           # save_data(results[3], "feature_importances")
        
       # if len(results) > 4:
          #  save_data(results[4], "results_summary")

        # Handle plots
      #  if len(results) > 5:
           # handle_plot(results[5], "train_plot")
        
      #  if len(results) > 6:
          #  handle_plot(results[6], "val_plot")
        
        if len(results) > 7:
            handle_plot(results[7], "test_plot")

        # Save additional CSV files
        if len(results) > 8:
            save_data(results[8], "train_additional")
        
        if len(results) > 9:
            save_data(results[9], "val_additional")
        
        if len(results) > 10:
            save_data(results[10], "test_additional")

        print("\nProcessing complete. Results saved in 'output' directory.")
        print("Contents of output directory:")
        print(os.listdir("output"))

    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
