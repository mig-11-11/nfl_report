# ============================================================
# NFL PBP ANALYSIS
# 2020-2025 REGULAR SEASON
#
# Python statistical analysis of SQL-generated datasets
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "exported_sql_data"
OUTPUT_DIR = "analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):
    """Load a CSV from the data directory."""
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"WARNING: Could not find {path}")
        return None

    df = pd.read_csv(path)

    print(f"Loaded {filename}: {df.shape[0]:,} rows")

    return df

def correlation_analysis(df, x_columns, target="win_pct"):
    """
    Calculate Pearson and Spearman correlations
    between variables and the target.
    """

    results = []

    for column in x_columns:

        temp = df[[column, target]].dropna()

        if len(temp) < 3:
            continue

        pearson_r, pearson_p = pearsonr(
            temp[column],
            temp[target]
        )

        spearman_r, spearman_p = spearmanr(
            temp[column],
            temp[target]
        )

        results.append({
            "variable": column,
            "n": len(temp),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p
        })

    return pd.DataFrame(results)


def regression_analysis(df, predictors, target="win_pct"):
    """
    Run OLS regression:
        target ~ predictors
    """

    temp = df[predictors + [target]].dropna()

    X = temp[predictors]
    y = temp[target]

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model


def save_regression_summary(model, filename):
    """Save regression summary to a text file."""

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w") as f:
        f.write(model.summary().as_text())


def scatter_plot(
    df,
    x,
    y,
    xlabel,
    ylabel,
    title,
    filename
):
    """Create and save a scatter plot."""

    temp = df[[x, y]].dropna()

    plt.figure(figsize=(9, 6))

    plt.scatter(
        temp[x],
        temp[y],
        alpha=0.7
    )

    # Regression line
    if len(temp) >= 2:

        coefficients = np.polyfit(
            temp[x],
            temp[y],
            1
        )

        x_line = np.linspace(
            temp[x].min(),
            temp[x].max(),
            100
        )

        y_line = (
            coefficients[0] * x_line
            + coefficients[1]
        )

        plt.plot(
            x_line,
            y_line
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.close()

# ============================================================
# LOAD DATA
# ============================================================

q1 = load_csv("red_zone.csv")

q2 = load_csv("epa_success.csv")

q3 = load_csv("explosive_plays.csv")

q4a = load_csv("qb_efficient.csv")

q4b = load_csv("rb_efficient.csv")

q4c = load_csv("wr_efficient.csv")

q5 = load_csv("qb_team_success.csv")

q6 = load_csv("third_down.csv")

q7 = load_csv("early_game.csv")

q8 = load_csv("outperform_efficiency.csv")

# ============================================================
# QUESTION 1
#
# Does red-zone efficiency translate into wins?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 1: RED-ZONE EFFICIENCY")
print("=" * 70)


if q1 is not None:

    variables = [
        "red_zone_td_rate",
        "red_zone_scoring_rate",
        "red_zone_turnover_rate",
        "red_zone_net_efficiency",
        "red_zone_epa_per_possession",
        "red_zone_epa_per_play"
    ]

    q1_correlations = correlation_analysis(
        q1,
        variables
    )

    print("\nCorrelations with win percentage:")
    print(
        q1_correlations.sort_values(
            "pearson_r",
            ascending=False
        ).to_string(index=False)
    )

    q1_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q1_correlations.csv"
        ),
        index=False
    )

    # Main regression
    predictors = [
        "red_zone_td_rate",
        "red_zone_turnover_rate",
        "red_zone_epa_per_play"
    ]

    model = regression_analysis(
        q1,
        predictors
    )

    print("\nRegression:")
    print(model.summary())

    save_regression_summary(
        model,
        "q1_regression.txt"
    )

    scatter_plot(
        q1,
        "red_zone_td_rate",
        "win_pct",
        "Red-Zone TD Rate",
        "Win Percentage",
        "Red-Zone TD Rate vs Win Percentage",
        "q1_red_zone_td_rate.png"
    )

    # ============================================================
# QUESTION 2
#
# Does EPA outperform traditional statistics?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 2: EPA VS TRADITIONAL STATISTICS")
print("=" * 70)


if q2 is not None:

    variables = [
        "yards_per_play",
        "epa_per_play",
        "success_rate",
        "turnover_rate"
    ]

    q2_correlations = correlation_analysis(
        q2,
        variables
    )

    print("\nCorrelations with win percentage:")
    print(
        q2_correlations.sort_values(
            "pearson_r",
            ascending=False
        ).to_string(index=False)
    )

    q2_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q2_correlations.csv"
        ),
        index=False
    )

    # Traditional statistics only
    traditional_model = regression_analysis(
        q2,
        [
            "yards_per_play",
            "success_rate",
            "turnover_rate"
        ]
    )

    # EPA model
    epa_model = regression_analysis(
        q2,
        [
            "epa_per_play",
            "turnover_rate"
        ]
    )

    print("\nTraditional model:")
    print(traditional_model.summary())

    print("\nEPA model:")
    print(epa_model.summary())

    save_regression_summary(
        traditional_model,
        "q2_traditional_model.txt"
    )

    save_regression_summary(
        epa_model,
        "q2_epa_model.txt"
    )

    # Compare R-squared
    comparison = pd.DataFrame({
        "model": [
            "Traditional",
            "EPA"
        ],
        "r_squared": [
            traditional_model.rsquared,
            epa_model.rsquared
        ],
        "adjusted_r_squared": [
            traditional_model.rsquared_adj,
            epa_model.rsquared_adj
        ]
    })

    print("\nModel comparison:")
    print(comparison)

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q2_model_comparison.csv"
        ),
        index=False
    )

    scatter_plot(
        q2,
        "epa_per_play",
        "win_pct",
        "EPA per Play",
        "Win Percentage",
        "EPA per Play vs Win Percentage",
        "q2_epa_vs_wins.png"
    )


# ============================================================
# QUESTION 3
#
# Do explosive plays contribute more than overall efficiency?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 3: EXPLOSIVE PLAYS")
print("=" * 70)


if q3 is not None:

    variables = [
        "explosive_play_rate",
        "epa_per_play",
        "yards_per_play"
    ]

    q3_correlations = correlation_analysis(
        q3,
        variables
    )

    print("\nCorrelations with win percentage:")
    print(
        q3_correlations.sort_values(
            "pearson_r",
            ascending=False
        ).to_string(index=False)
    )

    q3_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q3_correlations.csv"
        ),
        index=False
    )

    model = regression_analysis(
        q3,
        [
            "explosive_play_rate",
            "epa_per_play"
        ]
    )

    print("\nRegression:")
    print(model.summary())

    save_regression_summary(
        model,
        "q3_regression.txt"
    )

    scatter_plot(
        q3,
        "explosive_play_rate",
        "win_pct",
        "Explosive Play Rate",
        "Win Percentage",
        "Explosive Play Rate vs Win Percentage",
        "q3_explosive_plays.png"
    )


# ============================================================
# QUESTION 4
#
# Player efficiency
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 4: PLAYER EFFICIENCY")
print("=" * 70)


# ------------------------------------------------------------
# Q4A: QUARTERBACKS
# ------------------------------------------------------------

if q4a is not None:

    print("\nTop QBs by EPA per attempt:")

    top_qbs = (
        q4a
        .sort_values(
            "epa_per_attempt",
            ascending=False
        )
        .head(20)
    )

    print(
        top_qbs[
            [
                "season",
                "player",
                "pass_attempts",
                "epa_per_attempt",
                "cpoe",
                "success_rate"
            ]
        ].to_string(index=False)
    )

    top_qbs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q4_top_qbs.csv"
        ),
        index=False
    )


# ------------------------------------------------------------
# Q4B: RUNNING BACKS
# ------------------------------------------------------------

if q4b is not None:

    print("\nTop RBs by EPA per rush:")

    top_rbs = (
        q4b
        .sort_values(
            "epa_per_rush",
            ascending=False
        )
        .head(20)
    )

    print(
        top_rbs[
            [
                "season",
                "player",
                "carries",
                "epa_per_rush",
                "success_rate",
                "yards_per_rush"
            ]
        ].to_string(index=False)
    )

    top_rbs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q4_top_rbs.csv"
        ),
        index=False
    )


# ------------------------------------------------------------
# Q4C: RECEIVERS
# ------------------------------------------------------------

if q4c is not None:

    print("\nTop receivers by EPA per target:")

    top_receivers = (
        q4c
        .sort_values(
            "epa_per_target",
            ascending=False
        )
        .head(20)
    )

    print(
        top_receivers[
            [
                "season",
                "player",
                "targets",
                "catch_rate",
                "epa_per_target",
                "yards_per_reception"
            ]
        ].to_string(index=False)
    )

    top_receivers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q4_top_receivers.csv"
        ),
        index=False
    )


# ============================================================
# QUESTION 5
#
# Which QB statistics are most associated with team success?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 5: QB STATISTICS VS TEAM SUCCESS")
print("=" * 70)


if q5 is not None:

    variables = [
        "qb_epa_per_play",
        "qb_cpoe",
        "qb_success_rate"
    ]

    q5_correlations = correlation_analysis(
        q5,
        variables
    )

    print("\nCorrelations with win percentage:")

    print(
        q5_correlations.sort_values(
            "pearson_r",
            ascending=False
        ).to_string(index=False)
    )

    q5_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q5_correlations.csv"
        ),
        index=False
    )

    model = regression_analysis(
        q5,
        [
            "qb_epa_per_play",
            "qb_cpoe",
            "qb_success_rate"
        ]
    )

    print("\nRegression:")
    print(model.summary())

    save_regression_summary(
        model,
        "q5_regression.txt"
    )

    scatter_plot(
        q5,
        "qb_epa_per_play",
        "win_pct",
        "QB EPA per Play",
        "Win Percentage",
        "QB EPA per Play vs Win Percentage",
        "q5_qb_epa.png"
    )


# ============================================================
# QUESTION 6
#
# How important is third-down efficiency?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 6: THIRD-DOWN EFFICIENCY")
print("=" * 70)


if q6 is not None:

    variables = [
        "third_down_conversion_rate"
    ]

    q6_correlations = correlation_analysis(
        q6,
        variables
    )

    print("\nCorrelation with win percentage:")

    print(
        q6_correlations.to_string(
            index=False
        )
    )

    q6_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q6_correlations.csv"
        ),
        index=False
    )

    model = regression_analysis(
        q6,
        [
            "third_down_conversion_rate"
        ]
    )

    print("\nRegression:")
    print(model.summary())

    save_regression_summary(
        model,
        "q6_regression.txt"
    )

    scatter_plot(
        q6,
        "third_down_conversion_rate",
        "win_pct",
        "Third-Down Conversion Rate",
        "Win Percentage",
        "Third-Down Efficiency vs Win Percentage",
        "q6_third_down.png"
    )


# ============================================================
# QUESTION 7
#
# Does first-half performance predict the eventual winner?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 7: FIRST-HALF PERFORMANCE")
print("=" * 70)


if q7 is not None:

    # Correlation
    temp = q7[
        [
            "first_half_epa_per_play",
            "won_game"
        ]
    ].dropna()

    pearson_r, pearson_p = pearsonr(
        temp["first_half_epa_per_play"],
        temp["won_game"]
    )

    spearman_r, spearman_p = spearmanr(
        temp["first_half_epa_per_play"],
        temp["won_game"]
    )

    q7_results = pd.DataFrame({
        "metric": [
            "Pearson",
            "Spearman"
        ],
        "correlation": [
            pearson_r,
            spearman_r
        ],
        "p_value": [
            pearson_p,
            spearman_p
        ]
    })

    print("\nFirst-half EPA vs winning:")

    print(q7_results)

    q7_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q7_correlations.csv"
        ),
        index=False
    )

    # Logistic regression
    X = q7[
        ["first_half_epa_per_play"]
    ].dropna()

    y = q7.loc[
        X.index,
        "won_game"
    ]

    X = sm.add_constant(X)

    logistic_model = sm.Logit(
        y,
        X
    ).fit(disp=False)

    print("\nLogistic regression:")
    print(logistic_model.summary())

    with open(
        os.path.join(
            OUTPUT_DIR,
            "q7_logistic_regression.txt"
        ),
        "w"
    ) as f:

        f.write(
            logistic_model.summary().as_text()
        )

    # Probability of winning by EPA quartile
    q7["epa_quartile"] = pd.qcut(
        q7["first_half_epa_per_play"],
        4,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4"
        ]
    )

    quartile_results = (
        q7
        .groupby("epa_quartile", observed=True)
        ["won_game"]
        .agg(
            games="count",
            wins="sum",
            win_rate="mean"
        )
        .reset_index()
    )

    print("\nWin rate by first-half EPA quartile:")
    print(quartile_results)

    quartile_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q7_epa_quartiles.csv"
        ),
        index=False
    )


# ============================================================
# QUESTION 8
#
# Which teams outperform their underlying efficiency?
# ============================================================

print("\n")
print("=" * 70)
print("QUESTION 8: UNDERLYING EFFICIENCY")
print("=" * 70)


if q8 is not None:

    variables = [
        "offensive_epa_per_play",
        "offensive_yards_per_play",
        "offensive_success_rate",
        "offensive_turnover_rate",
        "defensive_epa_per_play",
        "defensive_yards_per_play",
        "defensive_success_rate"
    ]

    q8_correlations = correlation_analysis(
        q8,
        variables
    )

    print("\nCorrelations with win percentage:")

    print(
        q8_correlations.sort_values(
            "pearson_r",
            ascending=False
        ).to_string(index=False)
    )

    q8_correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q8_correlations.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Build an underlying-efficiency model
    # --------------------------------------------------------

    predictors = [
        "offensive_epa_per_play",
        "offensive_success_rate",
        "offensive_turnover_rate",
        "defensive_epa_per_play",
        "defensive_success_rate"
    ]

    model = regression_analysis(
        q8,
        predictors
    )

    print("\nUnderlying efficiency regression:")
    print(model.summary())

    save_regression_summary(
        model,
        "q8_efficiency_regression.txt"
    )

    # --------------------------------------------------------
    # Expected vs actual win percentage
    # --------------------------------------------------------

    temp = q8[predictors + ["win_pct"]].dropna()

    X = sm.add_constant(
        temp[predictors]
    )

    y = temp["win_pct"]

    model = sm.OLS(
        y,
        X
    ).fit()

    q8.loc[
        temp.index,
        "expected_win_pct"
    ] = model.predict(X)

    q8.loc[
        temp.index,
        "overperformance"
    ] = (
        q8.loc[
            temp.index,
            "win_pct"
        ]
        -
        q8.loc[
            temp.index,
            "expected_win_pct"
        ]
    )

    # Largest overperformers
    overperformers = (
        q8
        .sort_values(
            "overperformance",
            ascending=False
        )
        .head(20)
    )

    print("\nLargest overperformers:")

    print(
        overperformers[
            [
                "season",
                "team",
                "win_pct",
                "expected_win_pct",
                "overperformance"
            ]
        ].to_string(index=False)
    )

    overperformers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q8_overperformers.csv"
        ),
        index=False
    )

    # Largest underperformers
    underperformers = (
        q8
        .sort_values(
            "overperformance",
            ascending=True
        )
        .head(20)
    )

    print("\nLargest underperformers:")

    print(
        underperformers[
            [
                "season",
                "team",
                "win_pct",
                "expected_win_pct",
                "overperformance"
            ]
        ].to_string(index=False)
    )

    underperformers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q8_underperformers.csv"
        ),
        index=False
    )

    # Full expected vs actual dataset
    q8[
        [
            "season",
            "team",
            "win_pct",
            "expected_win_pct",
            "overperformance"
        ]
    ].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "q8_expected_vs_actual.csv"
        ),
        index=False
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

# Q1: Red-zone EPA/play vs win percentage
if q1 is not None:
    scatter_plot(q1, 'red_zone_epa_per_play', 'win_pct',
                 'Red-Zone EPA per Play', 'Win Percentage',
                 'Red-Zone EPA per Play vs Win Percentage',
                 'q1_red_zone_epa.png')

# Q2: Model R-squared comparison
if q2 is not None:
    plt.figure(figsize=(7, 5))
    values = [traditional_model.rsquared, epa_model.rsquared]
    bars = plt.bar(['Traditional\nStatistics', 'EPA +\nTurnovers'], values)
    plt.ylabel('R²')
    plt.title('Explanatory Power of Team-Success Models')
    plt.ylim(0, 0.65)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + .015,
                 f'{value:.3f}', ha='center')
    plt.grid(axis='y', alpha=.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q2_model_r2.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Q4: Top player efficiency
def player_efficiency_plot(df, metric, xlabel, title, filename):
    top = df.nlargest(8, metric).sort_values(metric)
    plt.figure(figsize=(8, 5.5))
    plt.barh(top['season'].astype(str) + ' ' + top['player'], top[metric])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis='x', alpha=.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

if q4a is not None:
    player_efficiency_plot(q4a, 'epa_per_attempt', 'EPA per Pass Attempt',
                           'Top Quarterback Seasons by EPA per Attempt', 'q4_top_qbs.png')
if q4b is not None:
    player_efficiency_plot(q4b, 'epa_per_rush', 'EPA per Rush',
                           'Top Running Back Seasons by EPA per Rush', 'q4_top_rbs.png')
if q4c is not None:
    player_efficiency_plot(q4c, 'epa_per_target', 'EPA per Target',
                           'Top Receiver Seasons by EPA per Target', 'q4_top_receivers.png')

# Q5: QB EPA/play vs win percentage
if q5 is not None:
    scatter_plot(q5, 'qb_epa_per_play', 'win_pct',
                 'QB EPA per Play', 'Win Percentage',
                 'QB EPA per Play vs Win Percentage', 'q5_qb_epa.png')

# Q6: Third-down conversion vs win percentage
if q6 is not None:
    scatter_plot(q6, 'third_down_conversion_rate', 'win_pct',
                 'Third-Down Conversion Rate', 'Win Percentage',
                 'Third-Down Efficiency vs Win Percentage', 'q6_third_down.png')

# Q7: First-half EPA quartiles vs win rate
if q7 is not None:
    temp = q7.copy()
    temp['epa_quartile'] = pd.qcut(temp['first_half_epa_per_play'], 4,
                                   labels=['Q1', 'Q2', 'Q3', 'Q4'])
    win_rates = temp.groupby('epa_quartile', observed=True)['won_game'].mean()
    plt.figure(figsize=(7, 5))
    bars = plt.bar(win_rates.index.astype(str), win_rates)
    plt.xlabel('First-Half EPA/Play Quartile')
    plt.ylabel('Win Rate')
    plt.title('Eventual Win Rate by First-Half EPA Quartile')
    plt.ylim(0, .85)
    for bar, value in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, value + .02,
                 f'{value:.1%}', ha='center')
    plt.grid(axis='y', alpha=.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q7_first_half_epa_quartiles.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# Q8: Actual vs efficiency-expected win percentage
if q8 is not None:
    temp = q8[predictors + ['win_pct']].dropna()
    X = sm.add_constant(temp[predictors])
    model_plot = sm.OLS(temp['win_pct'], X).fit()
    expected = model_plot.predict(X)
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(expected, temp['win_pct'], alpha=.65)
    low = min(expected.min(), temp['win_pct'].min())
    high = max(expected.max(), temp['win_pct'].max())
    plt.plot([low, high], [low, high], linewidth=2)
    plt.xlabel('Expected Win Percentage')
    plt.ylabel('Actual Win Percentage')
    plt.title('Actual vs Efficiency-Expected Win Percentage')
    plt.grid(alpha=.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'q8_actual_vs_expected.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nResults saved to: {OUTPUT_DIR}/")
print("\nGenerated:")
print("  - Correlation tables")
print("  - Regression summaries")
print("  - Player efficiency rankings")
print("  - Expected vs actual win analysis")
print("  - PNG visualizations")