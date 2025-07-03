# Load required libraries
library(rrBLUP)  # For A.mat()
library(sommer)   # For D.mat()

# ----------------------------------------
# STEP 1: Read and prepare genotype data
# ----------------------------------------
# Read CSV (assuming first column is Sample_ID)
geno_data <- read.csv("C:/Users/Ashmitha/Desktop/SNP.csv", 
                      header = TRUE, 
                      stringsAsFactors = FALSE)

# Extract sample IDs (optional, for tracking)
sample_ids <- geno_data[, 1]

# Convert to numeric matrix (exclude first column)
geno_matrix <- as.matrix(geno_data[, -1])  

# ----------------------------------------
# STEP 2: Handle missing data (IMPUTATION)
# ----------------------------------------
# Option 1: Simple mean imputation
geno_imputed <- apply(geno_matrix, 2, function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  return(x)
})


# Additive relationship matrix
additive_matrix <- A.mat(geno_imputed)

# Dominance relationship matrix
dominance_matrix <- D.mat(geno_imputed)

# (Optional) Set row/col names for reference
rownames(additive_matrix) <- sample_ids
colnames(additive_matrix) <- sample_ids
rownames(dominance_matrix) <- sample_ids
colnames(dominance_matrix) <- sample_ids

write.csv(additive_matrix, 
          "Additive2.csv", 
          row.names = FALSE)

write.csv(dominance_matrix, 
          "Dominance_DTF_hyderabad.csv", 
          row.names = FALSE)


 write.csv(additive_matrix, "Additive_Matrix.csv", row.names = TRUE)  # Adds empty first column in Excel