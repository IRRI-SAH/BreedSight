# =============================================================================
# Supplementary Tables S7 and S9
# Actual vs OCS-predicted yield & mid-parent BLUP vs observed yield
# =============================================================================

library(dplyr)

# -----------------------------------------------------------------------------
# INPUT: multi-location progeny data
# Required columns:
#   Location, Actual_GY, OCS_Predicted_GY, MidParent_BLUP
# -----------------------------------------------------------------------------
# pheno_df <- read.csv("multi_location_progeny_yields.csv")

# -----------------------------------------------------------------------------
# Table S7 – Comparative statistics (Actual vs OCS predicted)
# -----------------------------------------------------------------------------
Table_S7 <- pheno_df %>%
  group_by(Location) %>%
  summarise(
    Mean_Actual   = mean(Actual_GY, na.rm = TRUE),
    Mean_OCS      = mean(OCS_Predicted_GY, na.rm = TRUE),
    SD_Actual     = sd(Actual_GY, na.rm = TRUE),
    SD_OCS        = sd(OCS_Predicted_GY, na.rm = TRUE),
    Min_Actual    = min(Actual_GY, na.rm = TRUE),
    Min_OCS       = min(OCS_Predicted_GY, na.rm = TRUE),
    Q25_Actual    = quantile(Actual_GY, 0.25, na.rm = TRUE),
    Q25_OCS       = quantile(OCS_Predicted_GY, 0.25, na.rm = TRUE),
    Median_Actual = median(Actual_GY, na.rm = TRUE),
    Median_OCS    = median(OCS_Predicted_GY, na.rm = TRUE),
    Q75_Actual    = quantile(Actual_GY, 0.75, na.rm = TRUE),
    Q75_OCS       = quantile(OCS_Predicted_GY, 0.75, na.rm = TRUE),
    Max_Actual    = max(Actual_GY, na.rm = TRUE),
    Max_OCS       = max(OCS_Predicted_GY, na.rm = TRUE),
    .groups = "drop"
  )

print(Table_S7)
write.csv(Table_S7, "Table_S7.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Table S9 – Mid-parent BLUP vs observed yield
# -----------------------------------------------------------------------------
Table_S9 <- pheno_df %>%
  group_by(Location) %>%
  summarise(
    n            = sum(complete.cases(MidParent_BLUP, Actual_GY)),
    Pearson_r    = cor(MidParent_BLUP, Actual_GY, use = "complete.obs"),
    Spearman_rho = cor(MidParent_BLUP, Actual_GY, method = "spearman",
                       use = "complete.obs"),
    SD_midparent = sd(MidParent_BLUP, na.rm = TRUE),
    SD_observed  = sd(Actual_GY, na.rm = TRUE),
    Best_case_r  = SD_midparent / SD_observed,   # theoretical upper bound
    .groups = "drop"
  )

print(Table_S9)
write.csv(Table_S9, "Table_S9.csv", row.names = FALSE)

# -----------------------------------------------------------------------------
# Optional: Table S8 style agreement (OCS predicted vs actual)
# -----------------------------------------------------------------------------
Table_S8 <- pheno_df %>%
  group_by(Location) %>%
  summarise(
    n         = sum(complete.cases(OCS_Predicted_GY, Actual_GY)),
    Pearson_r = cor(OCS_Predicted_GY, Actual_GY, use = "complete.obs"),
    Spearman  = cor(OCS_Predicted_GY, Actual_GY, method = "spearman",
                    use = "complete.obs"),
    R2        = 1 - sum((OCS_Predicted_GY - Actual_GY)^2, na.rm = TRUE) /
                    sum((Actual_GY - mean(Actual_GY, na.rm = TRUE))^2, na.rm = TRUE),
    MAPE      = mean(abs(OCS_Predicted_GY - Actual_GY) / Actual_GY, na.rm = TRUE) * 100,
    Mean_bias = mean(OCS_Predicted_GY - Actual_GY, na.rm = TRUE),
    .groups = "drop"
  )

print(Table_S8)
write.csv(Table_S8, "Table_S8.csv", row.names = FALSE)

cat("\nTables written:\n  Table_S7.csv\n  Table_S9.csv\n  Table_S8.csv\n")
