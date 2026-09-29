from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tennis_stats.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load and investigate ----------
df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.columns)
print(df.describe())


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close()


# ---------- exploratory analysis ----------
def explore(feature, outcome):
    plt.scatter(df[feature], df[outcome], alpha=0.4)
    plt.xlabel(feature)
    plt.ylabel(outcome)
    plt.title(f"{feature} vs {outcome}")
    save_fig(f"explore_{feature}_vs_{outcome}")


# ---------- model helper ----------
def fit_and_score(features, outcome, plot=False):
    x = df[features]
    y = df[[outcome]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=6
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    if plot:
        y_predict = model.predict(x_test)
        plt.scatter(y_test, y_predict, alpha=0.4)
        plt.xlabel(f"Actual {outcome}")
        plt.ylabel(f"Predicted {outcome}")
        plt.title(f"{', '.join(features)} -> {outcome}")
        save_fig(f"model_{'_'.join(features)[:80]}_{outcome}")
    return score


def main():
    # Exploratory plots
    explore("ServiceGamesPlayed", "Winnings")
    explore("BreakPointsOpportunities", "Winnings")
    explore("Aces", "Winnings")
    explore("DoubleFaults", "Wins")
    explore("FirstServeReturnPointsWon", "Ranking")

    # Single-feature models
    print("\n--- Single feature ---")
    print("ServiceGamesPlayed -> Winnings:",
          fit_and_score(["ServiceGamesPlayed"], "Winnings", plot=True))

    single_feature_tests = [
        (["ReturnGamesPlayed"], "Winnings"),
        (["BreakPointsOpportunities"], "Winnings"),
        (["BreakPointsFaced"], "Winnings"),
        (["DoubleFaults"], "Winnings"),
        (["Aces"], "Winnings"),
        (["TotalPointsWon"], "Winnings"),
        (["ServiceGamesPlayed"], "Wins"),
        (["BreakPointsOpportunities"], "Ranking"),
    ]
    for features, outcome in single_feature_tests:
        print(features[0], "->", outcome, ":", fit_and_score(features, outcome))

    # Two-feature models
    print("\n--- Two features -> Winnings ---")
    two_feature_tests = [
        ["BreakPointsOpportunities", "ReturnGamesPlayed"],
        ["BreakPointsOpportunities", "ServiceGamesPlayed"],
        ["ReturnGamesPlayed", "ReturnGamesWon"],
        ["BreakPointsOpportunities", "Aces"],
        ["Aces", "DoubleFaults"],
    ]
    for features in two_feature_tests:
        print(features, fit_and_score(features, "Winnings"))

    # Multi-feature models
    print("\n--- Multiple features -> Winnings ---")
    service_features = [
        "Aces", "DoubleFaults", "FirstServe", "FirstServePointsWon",
        "SecondServePointsWon", "BreakPointsFaced", "BreakPointsSaved",
        "ServiceGamesPlayed", "ServiceGamesWon", "TotalServicePointsWon",
    ]
    return_features = [
        "FirstServeReturnPointsWon", "SecondServeReturnPointsWon",
        "BreakPointsOpportunities", "BreakPointsConverted",
        "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon",
        "TotalPointsWon",
    ]
    best_three = ["ReturnGamesPlayed", "BreakPointsOpportunities", "BreakPointsFaced"]

    multi_feature_tests = {
        "service only": service_features,
        "return only": return_features,
        "service + return": service_features + return_features,
        "best three": best_three,
    }
    for name, features in multi_feature_tests.items():
        print(name, "->", fit_and_score(features, "Winnings"))

    # Plot predictions for the best model
    fit_and_score(best_three, "Winnings", plot=True)


if __name__ == "__main__":
    main()













































































## perform multiple feature linear regressions here:
