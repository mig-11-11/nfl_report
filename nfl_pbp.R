# ============================================================
# NFL PLAY-BY-PLAY DATA: 2020-2025
# ============================================================

# Load packages
library(nflreadr)
library(readr)

seasons <- 2020:2025

# ------------------------------------------------------------
# 1. Load play-by-play data
# ------------------------------------------------------------

cat("Downloading NFL play-by-play data...\n")
cat("Seasons:", paste(seasons, collapse = ", "), "\n\n")

pbp <- load_pbp(seasons)

# ------------------------------------------------------------
# 2. Check the data
# ------------------------------------------------------------

cat("Download complete!\n\n")

cat("Number of rows:", nrow(pbp), "\n")
cat("Number of columns:", ncol(pbp), "\n\n")

cat("Seasons included:\n")
print(sort(unique(pbp$season)))

cat("\nGames included:", length(unique(pbp$game_id)), "\n")

# ------------------------------------------------------------
# 3. Save as CSV
# ------------------------------------------------------------

output_file <- "nfl_pbp_2020_2025.csv"

write_csv(
  pbp,
  output_file,
  na = ""
)

cat("\nSaved to:", normalizePath(output_file), "\n")

# ------------------------------------------------------------
# 4. Verify the file
# ------------------------------------------------------------

file_size_mb <- file.info(output_file)$size / 1024^2

cat("File size:", round(file_size_mb, 2), "MB\n")
cat("\nNFL PBP export complete!\n")