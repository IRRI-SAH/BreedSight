# =============================================================================
# SimpleMating – Optimal Cross Selection (BreedSight workflow)
# =============================================================================

# Install once:
# install.packages("devtools")
# devtools::install_github("Resende-Lab/SimpleMating")

library(SimpleMating)
library(dplyr)

# -----------------------------------------------------------------------------
# INPUTS (replace with your files)
# -----------------------------------------------------------------------------
# gebv_df  : data.frame with columns  ID, GEBV
# K        : genomic relationship matrix (rownames/colnames = IDs)
# type_df  : data.frame with columns  ID, Type  (I / P / T)

# Example loading:
# gebv_df <- read.csv("BreedSight_GEBVs.csv")
# K       <- as.matrix(read.csv("GRM.csv", row.names = 1))
# type_df <- read.csv("Accession_types.csv")

ids <- intersect(gebv_df$ID, rownames(K))
gebv_df <- gebv_df[match(ids, gebv_df$ID), ]
K       <- K[ids, ids]

Criterion <- data.frame(
  Id   = gebv_df$ID,
  Crit = gebv_df$GEBV
)

# -----------------------------------------------------------------------------
# 1. Generate possible crosses
# -----------------------------------------------------------------------------
MatePlan <- planCross(
  TargetPop  = Criterion$Id,
  MateDesign = "half"
)

# -----------------------------------------------------------------------------
# 2. Mid-parent value
# -----------------------------------------------------------------------------
ST_mpv <- getMPV(
  MatePlan  = MatePlan,
  Criterion = Criterion,
  K         = K
)

# -----------------------------------------------------------------------------
# 3. Usefulness criterion (UC)
#    UC = MPV + i * h * sigma_prog
# -----------------------------------------------------------------------------
Va <- 1.0          # replace with estimated additive variance
h  <- sqrt(0.25)   # replace with estimated narrow-sense heritability
i  <- 1.76         # selection intensity (top 10 %)

ST_mpv$sigma_prog <- sqrt((1 - ST_mpv$K) / 2 * Va)
ST_mpv$UC         <- ST_mpv$Y + i * h * ST_mpv$sigma_prog

cross_data <- ST_mpv %>%
  select(Parent1, Parent2, Y = UC, K)

# -----------------------------------------------------------------------------
# 4. Optional filters (as used in the paper)
# -----------------------------------------------------------------------------
type_lookup <- setNames(type_df$Type, type_df$ID)
cross_data$Type1 <- type_lookup[cross_data$Parent1]
cross_data$Type2 <- type_lookup[cross_data$Parent2]

cross_data <- cross_data %>%
  filter(
    !is.na(Type1), !is.na(Type2),
    Y >= 5000,
    K < 0.65
  )

# -----------------------------------------------------------------------------
# 5. Optimal cross selection
# -----------------------------------------------------------------------------
ocs_result <- selectCrosses(
  data                = cross_data,
  n.cross             = 566,
  max.cross           = 10,
  min.cross           = 1,
  culling.pairwise.k  = 0.65,
  max.cross.to.search = 1e7
)

mating_plan <- ocs_result[[2]]
write.csv(mating_plan, "OCS_selected_crosses.csv", row.names = FALSE)

cat("Selected crosses written to OCS_selected_crosses.csv\n")
head(mating_plan)
