"""
Statistical Evidence Metrics Library.
Computes rigorous quantitative metrics for backtesting:
- Wald-Wolfowitz Runs Test Z-score (Sequence Independence)
- Consecutive Wins & Losses Distribution
- High-Resolution Peak-to-Trough Drawdown & Duration
- Annualized Sharpe, Sortino, Calmar, and Deflated Sharpe Ratio (DSR)
- Win Rate, Payoff Ratio, Profit Factor, and Mathematical Expectancy
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class StatisticalEvidenceMetrics:
    """Institutional-grade statistical metrics for trade execution and equity curves."""

    @staticmethod
    def compute_runs_z_score(trade_pnl: List[float]) -> Dict[str, Any]:
        """
        Calculates the Wald-Wolfowitz Runs Test Z-Score.
        Tests the null hypothesis that winning and losing trades are statistically independent.
        """
        if len(trade_pnl) < 4:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "total_runs": 0,
                "expected_runs": 0.0,
                "variance": 0.0,
                "independent_at_95pct": True,
                "interpretation": "Insufficient trades for runs test",
            }

        # Binarize trades: +1 for Win (PnL > 0), -1 for Loss (PnL <= 0)
        signs = [1 if p > 0 else -1 for p in trade_pnl]
        n_wins = sum(1 for s in signs if s == 1)
        n_losses = sum(1 for s in signs if s == -1)
        n = len(signs)

        if n_wins == 0 or n_losses == 0:
            return {
                "z_score": 0.0,
                "p_value": 0.0,
                "total_runs": 1,
                "expected_runs": 1.0,
                "variance": 0.0,
                "independent_at_95pct": False,
                "interpretation": "Monolithic outcomes (all wins or all losses)",
            }

        # Count runs (groups of consecutive signs)
        runs = 1
        for i in range(1, n):
            if signs[i] != signs[i - 1]:
                runs += 1

        # Expected runs and variance under null hypothesis of independence
        mu_r = (2.0 * n_wins * n_losses) / n + 1.0
        var_r = (2.0 * n_wins * n_losses * (2.0 * n_wins * n_losses - n)) / (n**2 * (n - 1.0))

        if var_r <= 0:
            z_score = 0.0
            p_value = 1.0
        else:
            std_r = np.sqrt(var_r)
            z_score = (runs - mu_r) / std_r
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

        independent = abs(z_score) < 1.96

        if independent:
            interp = "Independent trade sequence (no streak clustering or Martingale vulnerability, 95% confidence)"
        elif z_score > 1.96:
            interp = "Excessive alternating outcomes (mean-reverting sequence, fewer streaks than random)"
        else:
            interp = "Clustered outcomes (trending streaks of wins/losses, potential regime dependence)"

        return {
            "z_score": float(z_score),
            "p_value": float(p_value),
            "total_runs": int(runs),
            "expected_runs": float(mu_r),
            "variance": float(var_r),
            "independent_at_95pct": bool(independent),
            "interpretation": interp,
        }

    @staticmethod
    def compute_streaks(trade_pnl: List[float]) -> Dict[str, Any]:
        """
        Computes consecutive wins and consecutive losses metrics.
        """
        if not trade_pnl:
            return {
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "avg_consecutive_wins": 0.0,
                "avg_consecutive_losses": 0.0,
                "win_streak_distribution": {},
                "loss_streak_distribution": {},
            }

        win_streaks = []
        loss_streaks = []

        curr_win = 0
        curr_loss = 0

        for pnl in trade_pnl:
            if pnl > 0:
                curr_win += 1
                if curr_loss > 0:
                    loss_streaks.append(curr_loss)
                    curr_loss = 0
            else:
                curr_loss += 1
                if curr_win > 0:
                    win_streaks.append(curr_win)
                    curr_win = 0

        if curr_win > 0:
            win_streaks.append(curr_win)
        if curr_loss > 0:
            loss_streaks.append(curr_loss)

        max_w = max(win_streaks) if win_streaks else 0
        max_l = max(loss_streaks) if loss_streaks else 0
        avg_w = float(np.mean(win_streaks)) if win_streaks else 0.0
        avg_l = float(np.mean(loss_streaks)) if loss_streaks else 0.0

        # Histogram distributions
        win_dist = {int(k): int(v) for k, v in pd.Series(win_streaks).value_counts().items()} if win_streaks else {}
        loss_dist = {int(k): int(v) for k, v in pd.Series(loss_streaks).value_counts().items()} if loss_streaks else {}

        return {
            "max_consecutive_wins": max_w,
            "max_consecutive_losses": max_l,
            "avg_consecutive_wins": round(avg_w, 2),
            "avg_consecutive_losses": round(avg_l, 2),
            "win_streak_distribution": win_dist,
            "loss_streak_distribution": loss_dist,
        }

    @staticmethod
    def compute_drawdowns(equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Computes peak-to-trough drawdowns, duration, and trailing profile.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_cash": 0.0,
                "avg_drawdown_pct": 0.0,
                "max_drawdown_duration_bars": 0,
                "max_drawdown_duration_time": "0s",
                "drawdown_series": pd.Series(dtype=float),
            }

        peak = equity_curve.cummax()
        dd_cash = peak - equity_curve
        dd_pct = (dd_cash / peak) * 100.0

        max_dd_pct = float(dd_pct.max())
        max_dd_cash = float(dd_cash.max())
        avg_dd_pct = float(dd_pct[dd_pct > 0].mean()) if (dd_pct > 0).any() else 0.0

        # Duration analysis: time spent under water
        underwater = dd_pct > 0
        durations = []
        curr_dur = 0
        for is_uw in underwater:
            if is_uw:
                curr_dur += 1
            else:
                if curr_dur > 0:
                    durations.append(curr_dur)
                    curr_dur = 0
        if curr_dur > 0:
            durations.append(curr_dur)

        max_dur_bars = max(durations) if durations else 0

        # Duration in time if index is datetime
        max_dur_time = "N/A"
        if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve.index) > 1:
            # Calculate time intervals
            median_delta = (equity_curve.index[1:] - equity_curve.index[:-1]).median()
            max_dur_time = str(median_delta * max_dur_bars)

        return {
            "max_drawdown_pct": round(max_dd_pct, 4),
            "max_drawdown_cash": round(max_dd_cash, 2),
            "avg_drawdown_pct": round(avg_dd_pct, 4),
            "max_drawdown_duration_bars": int(max_dur_bars),
            "max_drawdown_duration_time": max_dur_time,
            "drawdown_series": dd_pct,
        }

    @staticmethod
    def compute_risk_ratios(
        trade_returns: List[float],
        equity_curve: pd.Series,
        annualization_factor: float = 365.0 * 24.0 * 60.0,  # minute-bars in a crypto 24/7 year
    ) -> Dict[str, Any]:
        """
        Computes Sharpe, Sortino, Calmar, and Deflated Sharpe Ratio.
        """
        if len(trade_returns) < 2 or equity_curve.empty or len(equity_curve) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "deflated_sharpe_ratio": 0.0,
            }

        ret_series = pd.Series(trade_returns)
        mean_ret = float(ret_series.mean())
        std_ret = float(ret_series.std(ddof=1)) + 1e-9

        # Downside deviation
        neg_rets = ret_series[ret_series < 0]
        downside_std = float(neg_rets.std(ddof=1)) if len(neg_rets) > 1 else std_ret

        # Trade-level Sharpe & Sortino (annualized using trade frequency)
        trades_per_day = len(ret_series) / max(1.0, (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400.0) if isinstance(equity_curve.index, pd.DatetimeIndex) else 10.0
        annual_scale = np.sqrt(trades_per_day * 365.0)

        sharpe = (mean_ret / std_ret) * annual_scale
        sortino = (mean_ret / (downside_std + 1e-9)) * annual_scale

        # Calmar Ratio
        peak = equity_curve.cummax()
        max_dd_pct = float(((peak - equity_curve) / peak).max()) * 100.0
        total_return_pct = float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0) * 100.0)
        days = max(0.1, (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400.0) if isinstance(equity_curve.index, pd.DatetimeIndex) else 1.0
        ann_return_pct = total_return_pct * (365.0 / days)

        calmar = (ann_return_pct / max(max_dd_pct, 0.01))

        # Deflated Sharpe Ratio (Bailey & López de Prado)
        t_samples = len(trade_returns)
        skew = float(ret_series.skew()) if t_samples > 2 else 0.0
        kurt = float(ret_series.kurtosis()) if t_samples > 3 else 3.0

        n_eff = 10.0  # Effective number of trials
        sr_benchmark = np.sqrt(2.0 * np.log(n_eff) / max(t_samples, 2))

        sr_sample = mean_ret / std_ret
        numerator = (sr_sample - sr_benchmark) * np.sqrt(t_samples - 1.0)
        denom_var = 1.0 - skew * sr_sample + ((kurt - 1.0) / 4.0) * (sr_sample**2)
        denom = np.sqrt(max(denom_var, 1e-6))

        dsr_stat = numerator / denom
        dsr_prob = float(stats.norm.cdf(dsr_stat))

        return {
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "calmar_ratio": round(float(calmar), 2),
            "deflated_sharpe_ratio": round(float(dsr_prob), 4),
            "trade_skewness": round(skew, 3),
            "trade_kurtosis": round(kurt, 3),
        }

    @classmethod
    def evaluate_all(
        cls,
        trades_df: pd.DataFrame,
        equity_curve: pd.Series,
        initial_balance: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Master statistical evaluation synthesizing all metrics.
        """
        if trades_df.empty:
            return {"error": "No trades recorded in backtest."}

        pnl_list = trades_df["pnl_cash"].tolist()
        ret_list = trades_df["pnl_pct"].tolist()

        wins = trades_df[trades_df["pnl_cash"] > 0]
        losses = trades_df[trades_df["pnl_cash"] <= 0]

        n_trades = len(trades_df)
        n_wins = len(wins)
        n_losses = len(losses)

        win_rate = (n_wins / n_trades) * 100.0 if n_trades > 0 else 0.0
        loss_rate = (n_losses / n_trades) * 100.0 if n_trades > 0 else 0.0

        gross_profit = float(wins["pnl_cash"].sum()) if not wins.empty else 0.0
        gross_loss = float(abs(losses["pnl_cash"].sum())) if not losses.empty else 0.0
        net_profit = float(trades_df["pnl_cash"].sum())
        total_return_pct = (net_profit / initial_balance) * 100.0

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_win = float(wins["pnl_cash"].mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses["pnl_cash"].mean())) if not losses.empty else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Mathematical Expectancy
        expectancy_cash = (win_rate / 100.0 * avg_win) - (loss_rate / 100.0 * avg_loss)

        # Average holding time
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            holding_mins = (pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])).dt.total_seconds() / 60.0
            avg_holding_mins = float(holding_mins.mean())
        else:
            avg_holding_mins = 0.0

        # Sub-metrics
        streaks = cls.compute_streaks(pnl_list)
        runs_test = cls.compute_runs_z_score(pnl_list)
        drawdowns = cls.compute_drawdowns(equity_curve)
        risk_ratios = cls.compute_risk_ratios(ret_list, equity_curve)

        return {
            "initial_balance": initial_balance,
            "final_equity": round(initial_balance + net_profit, 2),
            "net_profit_cash": round(net_profit, 2),
            "total_return_pct": round(total_return_pct, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "total_trades": n_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate_pct": round(win_rate, 2),
            "loss_rate_pct": round(loss_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "payoff_ratio": round(payoff_ratio, 2),
            "expectancy_cash_per_trade": round(expectancy_cash, 2),
            "avg_holding_mins": round(avg_holding_mins, 1),
            "streaks": streaks,
            "runs_test": runs_test,
            "drawdowns": drawdowns,
            "risk_ratios": risk_ratios,
        }
