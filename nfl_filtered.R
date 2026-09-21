library(tidyverse)
library(nflreadr)

# ---------------------------------------------------------
# 1. Download NFL data
# ---------------------------------------------------------

seasons <- 2020:2025

pbp <- load_pbp(seasons)

# Player statistics
player_stats <- load_player_stats(seasons)

# Weekly player statistics
weekly_stats <- load_player_stats(seasons)

# Rosters
rosters <- load_rosters(seasons)

# Schedules
schedules <- load_schedules(seasons)


# ---------------------------------------------------------
# 2. Teams table
# ---------------------------------------------------------

teams <- schedules |>
  select(
    home_team,
    away_team
  ) |>
  pivot_longer(
    cols = c(home_team, away_team),
    values_to = "team",
    names_to = "team_side"
  ) |>
  select(team) |>
  filter(!is.na(team)) |>
  distinct() |>
  arrange(team)

print("Teams table being created.")
write_csv(teams, "/Users/matteo/Desktop/projects/nfl/teams.csv")


# ---------------------------------------------------------
# 3. Games table
# ---------------------------------------------------------

games <- schedules |>
  select(
    game_id,
    season,
    week,
    game_type,
    gameday,
    home_team,
    away_team,
    home_score,
    away_score,
    total,
    result,
    div_game
  ) |>
  distinct(game_id, .keep_all = TRUE)

print("Games table being created.")
write_csv(games, "/Users/matteo/Desktop/projects/nfl/games.csv")


# ---------------------------------------------------------
# 4. Team/Game statistics
# ---------------------------------------------------------

home_stats <- schedules |>
  select(
    game_id,
    season,
    week,
    home_team,
    away_team,
    home_score,
    away_score
  ) |>
  rename(
    team = home_team,
    opponent = away_team,
    points_for = home_score,
    points_against = away_score
  ) |>
  mutate(
    home_away = "HOME"
  )

away_stats <- schedules |>
  select(
    game_id,
    season,
    week,
    home_team,
    away_team,
    home_score,
    away_score
  ) |>
  rename(
    team = away_team,
    opponent = home_team,
    points_for = away_score,
    points_against = home_score
  ) |>
  mutate(
    home_away = "AWAY"
  )

team_game_stats <- bind_rows(
  home_stats,
  away_stats
) |>
  mutate(
    point_differential = points_for - points_against,
    win = case_when(
      points_for > points_against ~ 1,
      points_for < points_against ~ 0,
      TRUE ~ NA_real_
    )
  )

print("Team/Game statistics being created.")
write_csv(team_game_stats, "/Users/matteo/Desktop/projects/nfl/team_game_stats.csv")


# ---------------------------------------------------------
# 5. Clean player statistics
# ---------------------------------------------------------

player_game_stats <- player_stats |>
  select(
    season,
    week,
    player_id,
    player_name,
    team,
    position,
    fantasy_points,
    passing_yards,
    passing_tds,
    passing_interceptions,
    rushing_yards,
    rushing_tds,
    receptions,
    receiving_yards,
    receiving_tds
  ) |>
  mutate(
    across(
      c(
        fantasy_points,
        passing_yards,
        passing_tds,
        passing_interceptions,
        rushing_yards,
        rushing_tds,
        receptions,
        receiving_yards,
        receiving_tds
      ),
      ~ replace_na(.x, 0)
    )
  )

print("Player/Game statistics being created.")
write_csv(player_game_stats, "/Users/matteo/Desktop/projects/nfl/player_game_stats.csv")


# ---------------------------------------------------------
# 6. Player dimension table
# ---------------------------------------------------------

players <- player_game_stats |>
  select(
    player_id,
    player_name,
    position
  ) |>
  filter(!is.na(player_id)) |>
  distinct(player_id, .keep_all = TRUE)

print("Players table being created.")
write_csv(players, "/Users/matteo/Desktop/projects/nfl/players.csv")


# ---------------------------------------------------------
# 7. Season-level team performance
# ---------------------------------------------------------

team_season_summary <- team_game_stats |>
  group_by(season, team) |>
  summarise(
    games = n(),
    wins = sum(win, na.rm = TRUE),
    losses = sum(win == 0, na.rm = TRUE),
    points_for = sum(points_for, na.rm = TRUE),
    points_against = sum(points_against, na.rm = TRUE),
    avg_points_for = mean(points_for, na.rm = TRUE),
    avg_points_against = mean(points_against, na.rm = TRUE),
    avg_point_differential =
      mean(point_differential, na.rm = TRUE),
    .groups = "drop"
  )

print("Team/Season statistics being created.")
write_csv(
  team_season_summary,
  "/Users/matteo/Desktop/projects/nfl/team_season_summary.csv"
)


# ---------------------------------------------------------
# 8. Team offensive statistics
# ---------------------------------------------------------

offense <- pbp |>
  filter(
    !is.na(posteam),
    !is.na(yards_gained)
  ) |>
  group_by(
    season,
    game_id,
    posteam
  ) |>
  summarise(
    plays = n(),
    total_yards = sum(yards_gained, na.rm = TRUE),
    avg_yards_per_play =
      mean(yards_gained, na.rm = TRUE),

    pass_attempts =
      sum(pass_attempt == 1, na.rm = TRUE),

    completions =
      sum(complete_pass == 1, na.rm = TRUE),

    passing_yards =
      sum(yards_gained[pass_attempt == 1], na.rm = TRUE),

    rush_attempts =
      sum(rush_attempt == 1, na.rm = TRUE),

    rushing_yards =
      sum(yards_gained[rush_attempt == 1], na.rm = TRUE),

    touchdowns =
      sum(touchdown == 1, na.rm = TRUE),

    interceptions =
      sum(interception == 1, na.rm = TRUE),

    sacks =
      sum(sack == 1, na.rm = TRUE),

    turnovers =
      sum(interception == 1 | fumble_lost == 1, na.rm = TRUE),

    .groups = "drop"
  )


write_csv(
  offense,
  "/Users/matteo/Desktop/projects/nfl/team_game_offense.csv"
)

# ---------------------------------------------------------
# 9. Defensive statistics
# ---------------------------------------------------------

defense <- pbp %>%
  filter(
    !is.na(defteam),
    !is.na(yards_gained)
  ) %>%
  group_by(
    season,
    game_id,
    defteam
  ) %>%
  summarise(
    defensive_plays = n(),
    
    yards_allowed =
      sum(yards_gained, na.rm = TRUE),
    
    avg_yards_allowed =
      mean(yards_gained, na.rm = TRUE),
    
    sacks =
      sum(sack == 1, na.rm = TRUE),
    
    interceptions =
      sum(interception == 1, na.rm = TRUE),
    
    fumbles_recovered =
      sum(fumble_lost == 1, na.rm = TRUE),
    
    touchdowns_allowed =
      sum(touchdown == 1, na.rm = TRUE),
    
    .groups = "drop"
  )

write_csv(
  defense,
  "/Users/matteo/Desktop/projects/nfl/team_game_defense.csv"
)


