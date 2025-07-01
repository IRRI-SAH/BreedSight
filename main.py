import os
import pandas as pd
import matplotlib.pyplot as plt  # Import for handling plots
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

# Save predictions
os.makedirs("output", exist_ok=True)
train_pred.to_csv("output/train_predictions.csv", index=False)  # train_pred
val_pred.to_csv("output/val_predictions.csv", index=False)    # val_pred
test_pred.to_csv("output/test_predictions.csv", index=False)  # test_pred

# Save plots if they exist
if train_plot is not None:
    train_plot.figure.savefig("output/train_plot.png")
    plt.close(train_plot.figure)  # Close the figure to free memory
else:
    print("No training plot available")

if val_plot is not None:
    val_plot.figure.savefig("output/val_plot.png")
    plt.close(val_plot.figure)
else:
    print("No validation plot available")

if test_plot is not None:
    test_plot.figure.savefig("output/test_plot.png")
    plt.close(test_plot.figure)
else:
    print("No test plot available")