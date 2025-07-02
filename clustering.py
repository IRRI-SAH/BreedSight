# -*- coding: utf-8 -*-
"""
Created on Wed May 21 09:43:00 2025

@author: Ashmitha
"""

import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

def cluster_split_data(geno_file, n_clusters=10, test_size=0.2):
    # Load data
    geno = pd.read_csv(geno_file.name)
    sample_ids = geno.iloc[:, 0]
    geno_data = geno.iloc[:, 1:]  # Assume 4,000 SNPs
    
    # Preprocess
    geno_data = geno_data.fillna(geno_data.mode().iloc[0])  # Impute with mode
    
    # Dimensionality reduction (PCA)
    pca = PCA(n_components=0.95)  # Use 2 components for visualization
    geno_reduced = pca.fit_transform(geno_data)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=60, n_init=20)
    clusters = kmeans.fit_predict(geno_reduced)
    geno['Cluster'] = clusters
    
    # Create cluster plot with sample IDs
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(geno_reduced[:, 0], geno_reduced[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Genotypic Clusters (Hover points to see Sample IDs)')
    plt.colorbar(scatter, label='Cluster')
    
    # Annotate points with sample IDs (for interactive hover in Gradio)
    plot = plt.gcf()
    
    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=42)
    for train_idx, test_idx in sss.split(geno, geno['Cluster']):
        geno_train_df, geno_test_df = geno.iloc[train_idx], geno.iloc[test_idx]
    
    # Save outputs
    geno_train_df.to_csv("Training_data_Cluster.csv", index=False)
    geno_test_df.to_csv("Testing_data_Cluster.csv", index=False)
    
    # Create a cluster membership table for display
    cluster_table = geno[['SampleID', 'Cluster']].sort_values('Cluster')
    
    return (
        "Training_data_Cluster.csv",
        "Testing_data_Cluster.csv",
        plot,
        cluster_table
    )

# Gradio Interface
iface = gr.Interface(
    fn=cluster_split_data,
    inputs=[
        gr.File(label="Upload Genotypic Data (CSV)"),
        gr.Number(label="Number of Clusters", value=5, precision=0),
        gr.Slider(label="Test Size", minimum=0.1, maximum=0.8, value=0.8)
    ],
    outputs=[
        gr.File(label="Download Training Data"),
        gr.File(label="Download Testing Data"),
        gr.Plot(label="Cluster Visualization"),
        gr.Dataframe(label="Cluster Membership", headers=["Sample ID", "Cluster"])
    ],
    title="Genotypic Data Clustering Tool",
    description="Upload SNP data (CSV) to cluster samples and split into train/test sets."
)

iface.launch()