"""Risk analytics for daily return series.

This module is the quantitative core of the FinLearn dashboard: it does its own
risk/growth math (rolling beta vs. a benchmark, Sharpe ratio, maximum drawdown)
directly from stored daily returns — no vendor numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RiskCalculator:
    """Compute standard risk metrics from a DataFrame of daily returns.

    The input DataFrame must contain one row per trading day and at least one
    column of simple (arithmetic) daily returns. Optionally a second column of
    benchmark daily returns (e.g. S&P 500) enables beta estimation.

    Metrics provided
    ----------------
    * **30-day rolling beta** — covariance of asset/benchmark returns over a
      trailing 30-observation window divided by the benchmark's variance.
    * **Sharpe ratio** — mean excess return per unit of volatility, annualized
      with ``sqrt(252)``.
    * **Maximum drawdown** — largest peak-to-trough decline of the cumulative
      equity curve, reported as a negative fraction (e.g. ``-0.23`` = -23%).

    Parameters
    ----------
    asset_col : str, default ``"asset"``
        Column name holding the asset's daily returns.
    benchmark_col : str, default ``"benchmark"``
        Column name holding the benchmark's daily returns. May be absent;
        beta methods then raise or return ``NaN`` gracefully.
    risk_free_rate : float, default ``0.045``
        Annualized risk-free rate used to compute daily excess returns.
    periods_per_year : int, default ``252``
        Trading days per year used for annualization.
    window : int, default ``30``
        Trailing window length (in observations) for rolling beta.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> idx = pd.bdate_range("2025-01-01", periods=120)
    >>> bench = rng.normal(0.0004, 0.01, len(idx))
    >>> asset = 1.2 * bench + rng.normal(0.0006, 0.008, len(idx))
    >>> df = pd.DataFrame({"asset": asset, "benchmark": bench}, index=idx)
    >>> rc = RiskCalculator(df)
    >>> -1.0 <= rc.maximum_drawdown() <= 0.0
    True
    >>> np.isfinite(rc.sharpe_ratio())
    np.True_
    >>> len(rc.rolling_beta()) == len(df)
    True
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        asset_col: str = "asset",
        benchmark_col: str = "benchmark",
        risk_free_rate: float = 0.045,
        periods_per_year: int = 252,
        window: int = 30,
    ) -> None:
        """Initialize the calculator.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily-return frame indexed by date (or any monotonic index).
        asset_col : str
            Name of the asset return column.
        benchmark_col : str
            Name of the benchmark return column (optional in practice).
        risk_free_rate : float
            Annualized risk-free rate (e.g. 0.045 for 4.5%).
        periods_per_year : int
            Number of trading periods per year for annualization.
        window : int
            Rolling window size in observations (default 30 ≈ one month).

        Raises
        ------
        TypeError
            If ``returns`` is not a DataFrame.
        ValueError
            If ``asset_col`` is missing or ``window < 2``.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if asset_col not in returns.columns:
            raise ValueError(f"asset column {asset_col!r} not found in DataFrame")
        if window < 2:
            raise ValueError("window must be >= 2")

        self._df = returns.copy()
        self.asset_col = asset_col
        self.benchmark_col = benchmark_col
        self.risk_free_rate = float(risk_free_rate)
        self.periods_per_year = int(periods_per_year)
        self.window = int(window)

        # Clean view of the asset series: drop NaNs so downstream math is safe.
        self._asset: pd.Series = self._df[asset_col].dropna().astype(float)

    # ------------------------------------------------------------------ #
    # Basic statistics
    # ------------------------------------------------------------------ #
    @property
    def daily_returns(self) -> pd.Series:
        """Cleaned, NaN-dropped asset daily return series."""
        return self._asset

    def annualized_return(self) -> float:
        """Geometric mean return annualized over the sample.

        Returns
        -------
        float
            e.g. ``0.18`` for an 18% annualized compound growth rate.
        """
        r = self._asset.to_numpy()
        if r.size == 0:
            return float("nan")
        growth = float(np.prod(1.0 + r))
        if growth <= 0:
            return -1.0  # total loss
        return growth ** (self.periods_per_year / r.size) - 1.0

    def annualized_volatility(self) -> float:
        """Standard deviation of daily returns, annualized (sqrt-252 rule)."""
        r = self._asset.to_numpy()
        if r.size < 2:
            return float("nan")
        return float(np.std(r, ddof=1) * np.sqrt(self.periods_per_year))

    # ------------------------------------------------------------------ #
    # Rolling beta
    # ------------------------------------------------------------------ #
    def rolling_beta(self) -> pd.Series:
        """30-day (``self.window``) rolling beta versus the benchmark.

        Beta over each trailing window is computed as::

            beta_t = Cov(r_asset, r_bench) / Var(r_bench)

        Uses aligned inner-join of the two series so missing benchmark dates
        do not corrupt the windows.

        Returns
        -------
        pd.Series
            Rolling beta indexed by date; the first ``window - 1`` valid
            observations are ``NaN``. All ``NaN`` if no benchmark column.
        """
        if self.benchmark_col not in self._df.columns:
            return pd.Series(
                np.nan, index=self._asset.index, name=f"beta_{self.window}d"
            )

        paired = self._df[[self.asset_col, self.benchmark_col]].dropna()
        asset = paired[self.asset_col].astype(float)
        bench = paired[self.benchmark_col].astype(float)

        cov = asset.rolling(self.window).cov(bench)
        var = bench.rolling(self.window).var(ddof=1)
        beta = cov / var
        beta.name = f"beta_{self.window}d"
        return beta

    def current_beta(self) -> float:
        """Latest available rolling-beta value (float, or NaN if unavailable)."""
        b = self.rolling_beta()
        return float(b.iloc[-1]) if not b.empty and np.isfinite(b.iloc[-1]) else float("nan")

    # ------------------------------------------------------------------ #
    # Sharpe ratio
    # ------------------------------------------------------------------ #
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio of the full-sample daily returns.

        Computed as::

            SR = mean(r_daily - rf_daily) / std(r_daily - rf_daily) * sqrt(252)

        where ``rf_daily = (1 + risk_free_rate) ** (1/252) - 1``.

        Returns
        -------
        float
            The Sharpe ratio; ``NaN`` if fewer than 2 observations or zero
            dispersion.
        """
        r = self._asset.to_numpy()
        if r.size < 2:
            return float("nan")
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
        excess = r - rf_daily
        sd = float(np.std(excess, ddof=1))
        if sd == 0.0 or not np.isfinite(sd):
            return float("nan")
        return float(np.mean(excess) / sd * np.sqrt(self.periods_per_year))

    def rolling_sharpe(self) -> pd.Series:
        """Rolling annualized Sharpe ratio over ``self.window`` observations."""
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
        excess = self._asset - rf_daily
        sr = (
            excess.rolling(self.window).mean()
            / excess.rolling(self.window).std(ddof=1)
            * np.sqrt(self.periods_per_year)
        )
        sr.name = f"sharpe_{self.window}d"
        return sr

    # ------------------------------------------------------------------ #
    # Maximum drawdown
    # ------------------------------------------------------------------ #
    def drawdown_series(self) -> pd.Series:
        """Peak-to-current drawdown of the cumulative equity curve.

        Returns
        -------
        pd.Series
            Non-positive fractions (0 means at a new high).
        """
        r = self._asset
        wealth = (1.0 + r).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        dd.name = "drawdown"
        return dd

    def maximum_drawdown(self) -> float:
        """Worst peak-to-trough decline over the sample.

        Returns
        -------
        float
            A negative fraction, e.g. ``-0.32`` for a 32% max drawdown.
            ``0.0`` for an empty series.
        """
        dd = self.drawdown_series()
        return float(dd.min()) if not dd.empty else 0.0

    def max_drawdown_dates(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Return ``(peak_date, trough_date)`` of the worst drawdown episode."""
        if self._asset.empty:
            return (None, None)
        dd = self.drawdown_series()
        trough = dd.idxmin()
        wealth = (1.0 + self._asset).cumprod()
        peak = wealth.loc[:trough].idxmax()
        return (peak, trough)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, float]:
        """Bundle all headline metrics into a JSON-friendly dict.

        Returns
        -------
        dict[str, float]
            Keys: ``annualized_return``, ``annualized_volatility``,
            ``sharpe_ratio``, ``maximum_drawdown``, ``beta_30d``,
            ``observations``.
        """
        return {
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "maximum_drawdown": self.maximum_drawdown(),
            "beta_30d": self.current_beta(),
            "observations": float(self._asset.size),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RiskCalculator(asset={self.asset_col!r}, benchmark={self.benchmark_col!r}, "
            f"n={len(self._asset)}, window={self.window})"
        )
