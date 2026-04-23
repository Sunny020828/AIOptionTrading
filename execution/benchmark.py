import pandas as pd
import numpy as np
from backtest import summarize_mtm_path
import matplotlib.pyplot as plt


def generate_benchmark_signals(date, df, direction="long"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    day_data = df[df["Date"] == pd.to_datetime(date).normalize()]
    if day_data.empty:
        return []

    S = float(day_data["Close"].iloc[0])
    action = "BUY" if direction == "long" else "SELL"

    signals = [{
        "action": action,
        "contract": "BENCHMARK",
        "quantity": 1,
        "strategy": f"{direction}_benchmark",
        "leg": "benchmark_leg",
        "price_est": S,
    },
        {
            "meta": {
                "date": date,
                "scenario": f"{direction}_benchmark",
                "strategy_type": "benchmark",
                "underlying_price": S,
                "quantity": 1,
            }
        }]
    return signals


def calculate_benchmark_pnl(df, direction, start_date, end_date):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # 过滤区间
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    if df.empty:
        return None

    # 建仓价格（起始日收盘价）
    entry_price = df.loc[df["Date"] >= pd.to_datetime(start_date), "Close"].iloc[0]
    sign = 1 if direction == "long" else -1

    # 计算每日累计PnL
    df["Total_PnL"] = sign * (df["Close"] - entry_price)
    df["Daily_PnL"] = df["Total_PnL"].diff().fillna(df["Total_PnL"])

    return df


def plot_benchmark_pnl(pnl_df, direction):
    pnl_df = pnl_df.copy()
    pnl_df["Month"] = pnl_df["Date"].dt.to_period("M")
    monthly_pnl = pnl_df.groupby("Month")["portfolio_daily_pnl"].sum()

    monthly_pnl.plot(kind="bar", figsize=(12, 6), title=f"{direction.capitalize()} Benchmark Monthly PnL")
    plt.ylabel("PnL")
    plt.xlabel("Month")
    plt.show()

    return monthly_pnl



def calculate_benchmark_metrics(
    df,
    direction,
    start_date,
    end_date,
    account_size=1_000_000,
    allocation=0.1,
):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()

    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()
    if df.empty:
        return None, None

    df = df.sort_values("Date").reset_index(drop=True)

    # 如果同一天有很多行，只保留每天一个Close
    daily_price = (
        df.groupby("Date", as_index=False)["Close"]
        .first()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    entry_rows = daily_price.loc[daily_price["Date"] >= start_dt, "Close"]
    if entry_rows.empty:
        return None, None
    entry_price = float(entry_rows.iloc[0])

    sign = 1 if direction == "long" else -1
    invested_capital = account_size * allocation
    position_size = int(np.floor(invested_capital / entry_price))

    daily_price["portfolio_total_pnl"] = sign * (daily_price["Close"] - entry_price) * position_size
    daily_price["portfolio_daily_pnl"] = daily_price["portfolio_total_pnl"].diff().fillna(daily_price["portfolio_total_pnl"])
    daily_price["qty"] = 1

    metrics = summarize_mtm_path(
        daily_price,
        account_size=account_size,
        win_rate_mode="active",
        date_col="Date",
        qty_col="qty",
        daily_pnl_col="portfolio_daily_pnl",
        total_pnl_col="portfolio_total_pnl",
    )

    metrics.update({
        "account_size": float(account_size),
        "allocation": float(allocation),
        "invested_capital": float(invested_capital),
        "entry_price": float(entry_price),
        "position_size": float(position_size),
    })

    return daily_price, metrics

def load_benchmark_data(filename="merged_df.csv"):
    df = pd.read_csv(filename)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    return df




def plot_benchmark_curve(pnl_df, direction, year):
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_df["Date"], pnl_df["portfolio_total_pnl"], linewidth=2)

    plt.title(f"{direction.capitalize()} Benchmark PnL ({year})", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("PnL ($)", fontsize=14)

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df = load_benchmark_data(
        "/Users/ostrichzhang/Desktop/HKUST/25-26Spring/6000C/MFIT6000CAlgoTrading/data/merged_df.csv"
    )

    # 只看 long benchmark，按年度分别计算
    year_ranges = {
        "2024": ("2024-01-02", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
    }

    results = {}

    for year, (start_date, end_date) in year_ranges.items():
        benchmark_long_df, long_metrics = calculate_benchmark_metrics(
            df,
            direction="long",
            start_date=start_date,
            end_date=end_date,
        )

        results[year] = long_metrics

        # 画图
        if benchmark_long_df is not None:
            plot_benchmark_curve(benchmark_long_df, "long", year)

    # 打印年度结果
    for year, metrics in results.items():
        print(f"\n===== {year} long annual result =====")
        print(f"{'indicator':<25}{'Long':<20}")
        for key, val in metrics.items():
            print(f"{key:<25}{str(val):<20}")