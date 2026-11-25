###############################################################################
# Additive (G) and Dominance (D) genomic relationship matrices in base R
#
# Input : marker matrix M  (n individuals x m SNPs), coded 0 / 1 / 2
#         = number of copies of the counted allele. NA allowed.
# Output: G  additive  relationship matrix  (VanRaden 2008, method 1)
#         D  dominance relationship matrix  (Vitezica et al. 2013)
#
# Formulas (p = frequency of the counted allele, q = 1 - p)
#   Additive : Z = M - 2p                      G = Z Z' / (2 * sum(p q))
#   Dominance: W (per genotype class)
#                 genotype 2 ->  -2 q^2
#                 genotype 1 ->   2 p q
#                 genotype 0 ->  -2 p^2         D = W W' / sum((2 p q)^2)
#   Both are scaled so that the average diagonal is ~1 (under HWE).
###############################################################################

## ---------------------------------------------------------------- functions
prep_markers <- function(M, maf_min = 0.01, max_missing = 0.2) {
  M <- as.matrix(M)
  storage.mode(M) <- "double"
  if (!all(M %in% c(0, 1, 2, NA))) stop("Markers must be coded 0/1/2 (or NA).")

  # drop SNPs with too many missing values
  keep <- colMeans(is.na(M)) <= max_missing
  M <- M[, keep, drop = FALSE]

  # allele frequency of the counted allele (NA-aware)
  p <- colMeans(M, na.rm = TRUE) / 2

  # mean-impute missing genotypes (= 2p, so they contribute 0 to Z)
  na <- which(is.na(M), arr.ind = TRUE)
  if (nrow(na) > 0) M[na] <- 2 * p[na[, 2]]

  # drop monomorphic / rare SNPs
  keep <- pmin(p, 1 - p) >= maf_min
  list(M = M[, keep, drop = FALSE], p = p[keep])
}

# scale = "vanraden": ZZ' / (2*sum(pq))   -> diagonal ~ 1
# scale = "m"       : ZZ' / number_of_SNPs -> reproduces the BreedSight
#                     training_additive.csv / testing_additive.csv files exactly
additive_matrix <- function(M, p, Mnew = NULL, scale = c("vanraden", "m")) {
  scale <- match.arg(scale)
  Z <- sweep(M, 2, 2 * p, "-")                 # centre: M - 2p
  d <- if (scale == "vanraden") 2 * sum(p * (1 - p)) else ncol(M)
  if (is.null(Mnew)) tcrossprod(Z) / d         # n x n
  else tcrossprod(sweep(Mnew, 2, 2 * p, "-"), Z) / d   # new x train (use TRAIN p)
}

dominance_matrix <- function(M, p) {
  q <- 1 - p
  # build W with the three genotype classes (vectorised, column-wise)
  W <- matrix(0, nrow(M), ncol(M))
  for (j in seq_len(ncol(M))) {
    g <- M[, j]
    W[, j] <- ifelse(g < 0.5, -2 * p[j]^2,
              ifelse(g < 1.5,  2 * p[j] * q[j],
                              -2 * q[j]^2))
  }
  tcrossprod(W) / sum((2 * p * q)^2)           # WW' / sum((2pq)^2)
}

## ------------------------------------------------------------------- usage
# 1) Read your genotype file (rows = individuals, 0/1/2 columns).
#    Example for the BreedSight training file: id in col 1, phenotype in col 2,
#    SNPs v1..vm from col 3 on.
#
# dat <- read.csv("training_phenotypic_data.csv", check.names = FALSE)
# ids <- dat[[1]]
# M   <- dat[, -(1:2)]
#
# 2) Or a quick simulated example (runs as is):
set.seed(1)
n <- 200; m <- 1000
p_true <- runif(m, 0.05, 0.5)
M <- sapply(p_true, function(pp) rbinom(n, 2, pp))
M[sample(length(M), 500)] <- NA                # a few missing calls
ids <- paste0("ind", seq_len(n))

pm <- prep_markers(M, maf_min = 0.01)
G  <- additive_matrix(pm$M, pm$p)
D  <- dominance_matrix(pm$M, pm$p)
dimnames(G) <- dimnames(D) <- list(ids, ids)

## ------------------------------------------------------------------- checks
cat("SNPs used            :", ncol(pm$M), "\n")
cat("Mean diag(G)         :", round(mean(diag(G)), 3), "(expected ~1)\n")
cat("Mean diag(D)         :", round(mean(diag(D)), 3), "(expected ~1)\n")
cat("Mean off-diag(G), (D):", round(mean(G[upper.tri(G)]), 4),
    round(mean(D[upper.tri(D)]), 4), "(expected ~0)\n")
cat("G symmetric          :", isSymmetric(G), "\n")

## --------------------------------------------------------------- save
# Optional: make them positive definite for REML / GBLUP software
# G <- G + diag(1e-6, nrow(G));  D <- D + diag(1e-6, nrow(D))
write.csv(G, "additive_matrix_G.csv")
write.csv(D, "dominance_matrix_D.csv")

# Test-vs-train (cross) matrix: use the TRAINING allele frequencies p
#   Zt <- sweep(M_test, 2, 2 * p, "-");  Z <- sweep(M_train, 2, 2 * p, "-")
#   G_test_train <- tcrossprod(Zt, Z) / (2 * sum(p * (1 - p)))
