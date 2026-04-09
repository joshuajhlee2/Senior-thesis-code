import io
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize_scalar

PUT_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SP500_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
RF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"

START_DATE = "2007-01-01"
LOOKBACK = 252
RV_WINDOW = 21
BETA = 0.95
LAMBDA = 0.05
X_MIN = 0.05
X_MAX = 0.20
TURNOVER_PENALTY = 0.0025   
STATE_BANDWIDTH = 1.0
DATA_DIR = Path("data_cache")
REQUEST_TIMEOUT = 120
MAX_RETRIES = 5
BACKOFF = 2.0


def download_csv(url: str, cache_name: str | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / cache_name if cache_name else None

    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)

    headers = {"User-Agent": "Mozilla/5.0"}
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.Session() as sess:
                r = sess.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                text = r.text
                if cache_path is not None:
                    cache_path.write_text(text, encoding="utf-8")
                return pd.read_csv(io.StringIO(text))
        except requests.RequestException as e:
            last_err = e
            sleep_s = BACKOFF ** (attempt - 1)
            print(f"Download failed ({attempt}/{MAX_RETRIES}) for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(sleep_s)

    if cache_path is not None and cache_path.exists():
        print(f"Using stale cached file for {cache_name} after download failures.")
        return pd.read_csv(cache_path)

    raise RuntimeError(f"Could not download {url} after {MAX_RETRIES} attempts") from last_err


def load_cached_csv(cache_name: str) -> pd.DataFrame | None:
    cache_path = DATA_DIR / cache_name
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return None


def try_download_or_cache(url: str, cache_name: str, required: bool = True) -> pd.DataFrame | None:
    try:
        return download_csv(url, cache_name)
    except Exception as e:
        cached = load_cached_csv(cache_name)
        if cached is not None:
            print(f"Warning: download failed for {cache_name}; using cached copy.")
            return cached
        if required:
            raise
        print(f"Warning: could not obtain {cache_name}; continuing without it. Error: {e}")
        return None


def weighted_cvar(losses: np.ndarray, weights: np.ndarray, beta: float) -> float:
    order = np.argsort(losses)
    losses = losses[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    idx = min(np.searchsorted(cdf, beta), len(losses) - 1)
    var_beta = losses[idx]
    tail = losses >= var_beta - 1e-12
    if not np.any(tail):
        return float(var_beta)
    tail_weights = weights[tail]
    return float(np.sum(losses[tail] * tail_weights) / np.sum(tail_weights))


def annualized_return(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    years = len(r) / 252.0
    return float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(252))


def sharpe_ratio(r: pd.Series) -> float:
    vol = annualized_vol(r)
    return float((r.mean() * 252.0) / vol) if vol > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def tail_cvar_from_returns(r: pd.Series, beta: float) -> float:
    losses = -r.to_numpy()
    w = np.full(len(losses), 1.0 / len(losses))
    return weighted_cvar(losses, w, beta)


def gaussian_kernel_2d(x1: np.ndarray, x2: np.ndarray, bandwidth: float) -> np.ndarray:
    dist2 = np.sum((x1 - x2) ** 2, axis=1)
    w = np.exp(-0.5 * dist2 / (bandwidth ** 2))
    total = np.sum(w)
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


def prepare_data() -> pd.DataFrame:
    put = try_download_or_cache(PUT_URL, "put_history.csv", required=True).rename(columns={"DATE": "date", "PUT": "put_index"})
    put["date"] = pd.to_datetime(put["date"])
    put = put.sort_values("date")

    vix = try_download_or_cache(VIX_URL, "vix_history.csv", required=True).rename(columns={"DATE": "date", "CLOSE": "vix"})
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[["date", "vix"]].sort_values("date")

    spx_raw = try_download_or_cache(SP500_URL, "sp500_fred.csv", required=False)
    rf_raw = try_download_or_cache(RF_URL, "dgs3mo_fred.csv", required=False)

    df = put.merge(vix, on="date", how="inner")

    if spx_raw is not None:
        spx = spx_raw.rename(columns={"DATE": "date", "SP500": "sp500"})
        spx["date"] = pd.to_datetime(spx["date"])
        spx["sp500"] = pd.to_numeric(spx["sp500"], errors="coerce")
        spx = spx.dropna().sort_values("date")
        df = df.merge(spx, on="date", how="inner")
        df["spx_ret"] = df["sp500"].pct_change()
        df["rv_21d"] = df["spx_ret"].rolling(RV_WINDOW).std(ddof=1) * np.sqrt(252.0) * 100.0
        df["vrp_signal"] = df["vix"] - df["rv_21d"]
    else:
        df["spx_ret"] = np.nan
        df["rv_21d"] = np.nan
        df["vrp_signal"] = -df["vix"]
        print("Running in fallback mode: SP500 data unavailable, so vrp_signal is replaced with -VIX.")

    if rf_raw is not None:
        rf = rf_raw.rename(columns={"DATE": "date", "DGS3MO": "rf_annual_pct"})
        rf["date"] = pd.to_datetime(rf["date"])
        rf["rf_annual_pct"] = pd.to_numeric(rf["rf_annual_pct"], errors="coerce")
        rf = rf.sort_values("date")
        rf["rf_daily"] = (rf["rf_annual_pct"] / 100.0) / 252.0
        df = df.merge(rf[["date", "rf_daily"]], on="date", how="left")
        df["rf_daily"] = df["rf_daily"].ffill().fillna(0.0)
    else:
        df["rf_daily"] = 0.0
        print("Running with zero cash return because 3M Treasury data is unavailable.")

    df = df[df["date"] >= pd.Timestamp(START_DATE)].copy()
    df["risky_ret"] = df["put_index"].pct_change()

    roll = LOOKBACK
    df["vix_mean"] = df["vix"].rolling(roll).mean()
    df["vix_std"] = df["vix"].rolling(roll).std(ddof=1)
    df["vrp_mean"] = df["vrp_signal"].rolling(roll).mean()
    df["vrp_std"] = df["vrp_signal"].rolling(roll).std(ddof=1)
    df["vix_z"] = (df["vix"] - df["vix_mean"]) / df["vix_std"]
    df["vrp_z"] = (df["vrp_signal"] - df["vrp_mean"]) / df["vrp_std"]

    df = df.dropna(subset=["risky_ret", "vix", "vrp_signal", "vix_z", "vrp_z"]).reset_index(drop=True)
    return df


def optimize_exposure_continuous(risky: np.ndarray, cash: np.ndarray, weights: np.ndarray, prev_a: float) -> tuple[float, float, float, float]:
    def objective(a: float) -> float:
        port = a * risky + (1.0 - a) * cash
        mu = float(np.sum(weights * port))
        cvar = weighted_cvar(-port, weights, BETA)
        smooth = TURNOVER_PENALTY * (a - prev_a) ** 2
        # maximize mu - lambda*cvar - smooth  <=> minimize negative
        return -(mu - LAMBDA * cvar - smooth)

    res = minimize_scalar(objective, bounds=(X_MIN, X_MAX), method="bounded", options={"xatol": 1e-4, "maxiter": 200})
    a_star = float(np.clip(res.x, X_MIN, X_MAX))
    port = a_star * risky + (1.0 - a_star) * cash
    mu = float(np.sum(weights * port))
    cvar = weighted_cvar(-port, weights, BETA)
    obj = mu - LAMBDA * cvar - TURNOVER_PENALTY * (a_star - prev_a) ** 2
    return a_star, obj, mu, cvar


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposures, objectives, mus, cvars = [], [], [], []
    prev_a = X_MAX

    for i in range(len(df)):
        if i < LOOKBACK:
            exposures.append(X_MAX)
            objectives.append(np.nan)
            mus.append(np.nan)
            cvars.append(np.nan)
            prev_a = X_MAX
            continue

        hist = df.iloc[i - LOOKBACK:i].copy()
        current_state = np.array([df.loc[i, "vrp_z"], df.loc[i, "vix_z"]])
        hist_state = hist[["vrp_z", "vix_z"]].to_numpy()
        weights = gaussian_kernel_2d(hist_state, current_state, STATE_BANDWIDTH)
        risky = hist["risky_ret"].to_numpy()
        cash = hist["rf_daily"].to_numpy()

        best_a, best_obj, best_mu, best_cvar = optimize_exposure_continuous(risky, cash, weights, prev_a)
        exposures.append(best_a)
        objectives.append(best_obj)
        mus.append(best_mu)
        cvars.append(best_cvar)
        prev_a = best_a

    out = df.copy()
    out["exp_static"] = X_MAX
    out["exp_cvar"] = exposures
    out["obj_cvar"] = objectives
    out["mu_state"] = mus
    out["cvar_state"] = cvars

    out["ret_static"] = out["exp_static"] * out["risky_ret"] + (1.0 - out["exp_static"]) * out["rf_daily"]
    out["ret_cvar"] = out["exp_cvar"] * out["risky_ret"] + (1.0 - out["exp_cvar"]) * out["rf_daily"]

    for name in ["static", "cvar"]:
        out[f"wealth_{name}"] = (1.0 + out[f"ret_{name}"]).cumprod()
        out[f"dd_{name}"] = out[f"wealth_{name}"] / out[f"wealth_{name}"].cummax() - 1.0

    summary = pd.DataFrame(
        {
            "Strategy": ["Static", "CVaR-Optimized"],
            "Ann Return": [annualized_return(out["ret_static"]), annualized_return(out["ret_cvar"])],
            "Ann Vol": [annualized_vol(out["ret_static"]), annualized_vol(out["ret_cvar"])],
            "Sharpe": [sharpe_ratio(out["ret_static"]), sharpe_ratio(out["ret_cvar"])],
            "Max Drawdown": [max_drawdown(out["ret_static"]), max_drawdown(out["ret_cvar"])],
            f"CVaR {int(BETA * 100)}%": [tail_cvar_from_returns(out["ret_static"], BETA), tail_cvar_from_returns(out["ret_cvar"], BETA)],
            "Avg Exposure": [float(out["exp_static"].mean()), float(out["exp_cvar"].mean())],
            "Min Exposure": [float(out["exp_static"].min()), float(out["exp_cvar"].min())],
            "Max Exposure": [float(out["exp_static"].max()), float(out["exp_cvar"].max())],
        }
    )
    return out, summary


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    df.to_csv("strategy_timeseries_v6.csv", index=False)
    summary.to_csv("strategy_summary_v6.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["wealth_static"], label="Static")
    ax.plot(df["date"], df["wealth_cvar"], label="CVaR-Optimized")
    ax.set_title("Equity Curves: Static vs CVaR-Optimized")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("equity_curves_v6.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["exp_cvar"], label="CVaR Exposure")
    ax.axhline(X_MAX, linestyle="--", label="Static Exposure")
    ax.axhline(X_MIN, linestyle=":", label="CVaR Floor")
    ax.set_title("Exposure Policy Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("exposure_policy_v6.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["dd_static"], label="Static")
    ax.plot(df["date"], df["dd_cvar"], label="CVaR-Optimized")
    ax.set_title("Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("drawdowns_v6.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(df["vrp_signal"], df["exp_cvar"], c=df["vix"], s=10)
    ax.set_title("CVaR Exposure vs VRP Signal")
    ax.set_xlabel("VRP Signal = VIX - Realized Vol (annualized vol points)")
    ax.set_ylabel("Chosen Exposure")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("VIX")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("signal_scatter_v6.png", dpi=200)
    plt.close(fig)


def print_interpretation(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    s = summary.set_index("Strategy")
    unique_exp = df["exp_cvar"].round(4).nunique()
    print("\nSummary table:\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
    print(f"\nUnique CVaR exposures (rounded to 4 d.p.): {unique_exp}")
    print("Saved files: strategy_summary.csv, strategy_timeseries.csv, equity_curves.png, exposure_policy.png, drawdowns.png, signal_scatterplot.png")


def main() -> None:
    df = prepare_data()
    out, summary = run_backtest(df)
    save_outputs(out, summary)
    print_interpretation(summary, out)


if __name__ == "__main__":
    main()
