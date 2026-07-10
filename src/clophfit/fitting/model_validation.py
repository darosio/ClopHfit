"""Reusable model-validation helpers for ClopHfit fitting workflows.

These utilities are designed to live in :mod:`clophfit.fitting` and be reused by
both package tests and manuscript-analysis scripts.  They intentionally avoid any
manuscript-specific paths, file formats, or plate names.
"""

from __future__ import annotations

import copy
import itertools
import typing as _t
import warnings
from dataclasses import dataclass

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sp_stats

from clophfit.fitting.residuals import (
    BIAS_P_VALUE_THRESHOLD,
    DW_LOWER_BOUND,
    DW_UPPER_BOUND,
    OUTLIER_RATE_THRESHOLD,
    OUTLIER_THRESHOLD_2SIGMA,
    OUTLIER_THRESHOLD_3SIGMA,
)

ArrayLike = _t.Any

STUDENT_T_NU = 3.0

RESIDUAL_TABLE_COLUMNS = [
    "trace_id",
    "well",
    "label",
    "step",
    "raw_i",
    "x",
    "y",
    "yhat",
    "sigma",
    "raw_res",
    "likelihood_res",
    "std_res",
    "p_outlier_per_point",
    "residual_likelihood",
    "student_t_nu",
    "is_residual_outlier",
    "outlier_threshold",
]


@dataclass
class ResidualAnalysis:
    """Statistical and structural residual analyses over a residual table.

    The single home for frame-returning residual analyses: per-(trace, label)
    distribution stats (:meth:`distribution_summary`), residual-vs-x correlation
    (:meth:`x_correlation`), lag-1 autocorrelation (:meth:`lag1_autocorrelation`),
    covariance and correlation across x-positions (:meth:`covariance`,
    :meth:`correlation`), systematic label bias (:meth:`label_bias`), and boolean
    quality-control checks (:meth:`validate`). Reach it from
    :attr:`ResidualDiagnostics.analysis`, or construct it directly from a
    canonical residual table.

    Parameters
    ----------
    residuals : pd.DataFrame
        Canonical residual table (Schema B) carrying ``well``/``label``/``x``/
        ``std_res`` columns.
    """

    residuals: pd.DataFrame

    def distribution_summary(self) -> pd.DataFrame:
        """Per-(trace, label) residual distribution stats.

        See :func:`residual_distribution_summary`. Supersedes
        :func:`clophfit.fitting.residuals.residual_statistics`.
        """
        return residual_distribution_summary(self.residuals)

    def x_correlation(self) -> pd.DataFrame:
        """Pearson/Spearman correlation of residuals against x.

        See :func:`residual_x_correlation`. Relates to
        :func:`clophfit.fitting.residuals.estimate_x_shift_statistics`;
        :func:`residual_x_trend_summary` gives the by-step trend.
        """
        return residual_x_correlation(self.residuals)

    def lag1_autocorrelation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Lag-1 residual autocorrelation per well and its per-label summary.

        See :func:`residual_lag1_autocorrelation`. Supersedes
        :func:`clophfit.fitting.residuals.detect_adjacent_correlation`.
        """
        return residual_lag1_autocorrelation(self.residuals)

    def covariance(
        self, *, value_col: str = "std_res", by: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """Per-label covariance of residuals across titration points.

        Wells are aligned on a shared point index (``by``) rather than the raw
        ``x``, because in real titrations each well carries its own jittered
        x-values (e.g. pH), which would leave no two wells sharing a column.

        Parameters
        ----------
        value_col : str
            Residual column to use (default ``"std_res"``).
        by : str | None
            Column that indexes the shared titration point across wells.
            ``None`` auto-selects the first available of ``"step"``, ``"raw_i"``,
            ``"x"``.

        Returns
        -------
        dict[str, pd.DataFrame]
            One point-by-point covariance matrix per label, over the wells that
            share a complete set of points. Axes are labelled by the mean ``x``
            at each point when available. Labels with fewer than two complete
            wells map to an empty frame.
        """
        align = by or next(
            (c for c in ("step", "raw_i", "x") if c in self.residuals.columns), "x"
        )
        cov_by_label: dict[str, pd.DataFrame] = {}
        for lbl, g in self.residuals.groupby("label"):
            pivot = g.pivot_table(
                index="well", columns=align, values=value_col, aggfunc="mean"
            )
            # complete-case wells only, for a clean covariance across points
            pivot = pivot.dropna(axis=0, how="any")
            if pivot.shape[0] < 2 or pivot.shape[1] < 1:
                cov_by_label[str(lbl)] = pd.DataFrame()
                continue
            if align != "x" and "x" in g.columns:
                xmap = g.groupby(align)["x"].mean().round(3)
                axis = [xmap.get(k, k) for k in pivot.columns]
            else:
                axis = pivot.columns.to_list()
            cov = np.atleast_2d(
                np.cov(pivot.to_numpy(dtype=float), rowvar=False, ddof=1)
            )
            cov_by_label[str(lbl)] = pd.DataFrame(cov, index=axis, columns=axis)
        return cov_by_label

    def correlation(
        self, *, value_col: str = "std_res", by: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """Per-label correlation matrices derived from :meth:`covariance`.

        Parameters
        ----------
        value_col : str
            Residual column to use (default ``"std_res"``).
        by : str | None
            Shared point index forwarded to :meth:`covariance`.

        Returns
        -------
        dict[str, pd.DataFrame]
            One point-by-point correlation matrix per label (empty where the
            covariance is empty).
        """
        corr_by_label: dict[str, pd.DataFrame] = {}
        for lbl, cov_df in self.covariance(value_col=value_col, by=by).items():
            if cov_df.empty:
                corr_by_label[lbl] = cov_df
                continue
            cov = cov_df.to_numpy()
            std = np.sqrt(np.diag(cov))
            std_outer = np.outer(std, std)
            corr = np.divide(
                cov, std_outer, out=np.full_like(cov, np.nan), where=std_outer != 0
            )
            corr_by_label[lbl] = pd.DataFrame(
                corr, index=cov_df.index, columns=cov_df.columns
            )
        return corr_by_label

    def label_bias(self, *, n_bins: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Detect systematic residual bias by label and x-range bin.

        Residuals are globally z-scored before aggregation so the summaries are
        comparable across labels.

        Parameters
        ----------
        n_bins : int
            Number of equal-width x bins for the per-(label, bin) summary.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(bias_by_label_and_bin, bias_by_label)``.
        """
        outlier_threshold = OUTLIER_THRESHOLD_3SIGMA
        strong_negative_threshold = -0.5

        df = self.residuals.copy()
        df["x_bin"] = pd.cut(df["x"], bins=n_bins)
        mean_resid = df["std_res"].mean()
        std_resid = df["std_res"].std()
        df["std_res"] = (df["std_res"] - mean_resid) / std_resid

        bias_summary = df.groupby(["label", "x_bin"], observed=False).agg(
            mean_resid=("std_res", "mean"),
            std_resid=("std_res", "std"),
            count=("std_res", "count"),
            outlier_rate=("std_res", lambda x: (np.abs(x) > outlier_threshold).mean()),
            mean_std_res=("std_res", "mean"),
        )
        label_bias = df.groupby("label", observed=False).agg(
            mean_resid=("std_res", "mean"),
            std_resid=("std_res", "std"),
            median_resid=("std_res", "median"),
            outlier_rate=("std_res", lambda x: (np.abs(x) > outlier_threshold).mean()),
            negative_bias_frac=(
                "std_res",
                lambda x: (x < strong_negative_threshold).mean(),
            ),
        )
        return bias_summary, label_bias

    def validate(self, *, verbose: bool = False) -> dict[str, bool]:
        """Boolean residual-quality checks over the table.

        Runs three checks: per-label systematic bias (t-test against zero), the
        overall outlier rate (beyond ±2 sigma), and serial correlation within
        each label (Durbin-Watson statistic on x-sorted residuals).

        Parameters
        ----------
        verbose : bool
            Print a warning for each failed check.

        Returns
        -------
        dict[str, bool]
            ``{"bias_ok", "outliers_ok", "correlation_ok"}``. An empty table
            passes every check.
        """
        checks = {"bias_ok": True, "outliers_ok": True, "correlation_ok": True}
        df = self.residuals
        if df.empty or "std_res" not in df.columns:
            return checks

        # Check 1: systematic bias (t-test against 0) per label
        for lbl, group in df.groupby("label"):
            r_lbl = group["std_res"].to_numpy(dtype=float)
            if len(r_lbl) > 1:
                _t_stat, p_value = sp_stats.ttest_1samp(r_lbl, 0)
                if p_value < BIAS_P_VALUE_THRESHOLD:
                    checks["bias_ok"] = False
                    if verbose:
                        print(
                            f"⚠️  Systematic bias detected in label {lbl} "
                            f"(mean={r_lbl.mean():.3f}, p={p_value:.4f})"
                        )

        # Check 2: outliers (more than 5% beyond ±2-sigma)
        r_all = df["std_res"].to_numpy(dtype=float)
        outlier_rate = float((np.abs(r_all) > OUTLIER_THRESHOLD_2SIGMA).mean())
        checks["outliers_ok"] = bool(outlier_rate < OUTLIER_RATE_THRESHOLD)
        if not checks["outliers_ok"] and verbose:
            print(f"⚠️  High outlier rate: {outlier_rate:.1%} beyond ±2-sigma")

        # Check 3: serial correlation (Durbin-Watson) within labels
        ss_diff = 0.0
        ss_total = 0.0
        for _lbl, group in df.groupby("label"):
            g_sorted = group.sort_values("x")
            r_lbl = g_sorted["std_res"].to_numpy(dtype=float)
            if len(r_lbl) > 1:
                ss_diff += float(np.sum(np.diff(r_lbl) ** 2))
                ss_total += float(np.sum(r_lbl**2))
        if ss_total > 0:
            dw_stat = ss_diff / ss_total
            checks["correlation_ok"] = bool(DW_LOWER_BOUND < dw_stat < DW_UPPER_BOUND)
            if not checks["correlation_ok"] and verbose:
                print(f"⚠️  Serial correlation detected (DW={dw_stat:.2f})")
        return checks


@dataclass
class ResidualDiagnostics:
    """Single fluent home for residual analysis over a canonical residual table.

    Build one from a fit (``ResidualDiagnostics.from_fit_results(...)``) or wrap
    an existing table (``ResidualDiagnostics(fr.residuals)``), then chain
    transforms and read summaries. This is the place to reach for residual
    analysis; the module-level ``residual_*`` functions are its building blocks.

    Transforms (return a new ``ResidualDiagnostics``): :meth:`annotate`,
    :meth:`step_centered`, :meth:`label_scaled`, :meth:`well_scaled`,
    :meth:`with_relative_residuals`.

    Summaries (return frames): :meth:`well_summary`, :meth:`normality`,
    :meth:`step_summary`, :meth:`position_summary`, :meth:`tail_rows`.

    Statistical and structural analyses (distribution, x-correlation, lag-1
    autocorrelation, covariance, correlation, label bias, QA checks) live under
    :attr:`analysis` (a :class:`ResidualAnalysis`). Their module-level building
    blocks are ``residual_distribution_summary`` / ``residual_x_correlation`` /
    ``residual_lag1_autocorrelation`` / ``residual_x_trend_summary`` /
    ``residual_cross_label_correlation``.

    Plots: :meth:`plot_hist_qq`, :meth:`plot_step`, :meth:`plot_role`,
    :meth:`plot_col`, :meth:`plot_well_summary`.
    """

    residuals: pd.DataFrame
    value_col: str = "std_res"

    def __post_init__(self) -> None:
        """Copy the input residual table so chained helpers never mutate callers."""
        self.residuals = self.residuals.copy()

    @classmethod
    def from_fit_results(
        cls,
        results: dict[str, _t.Any],
        trace_id: str,
        binding_function: _t.Callable[..., ArrayLike],
        *,
        robust: bool = False,
        value_col: str = "std_res",
    ) -> ResidualDiagnostics:
        """Create diagnostics from a mapping of well IDs to FitResult objects."""
        return cls(
            residuals_from_fit_results(
                results,
                trace_id,
                binding_function,
                robust=robust,
            ),
            value_col=value_col,
        )

    def annotate(
        self,
        *,
        fit_df: pd.DataFrame | None = None,
        ctrl_wells: _t.Iterable[str] = (),
        extra_ctrl_wells: _t.Iterable[str] = (),
    ) -> ResidualDiagnostics:
        """Return diagnostics annotated with fit parameters, role, row, and column."""
        df = self.residuals.copy()
        df["row"] = df["well"].astype(str).str[0]
        df["col"] = pd.to_numeric(df["well"].astype(str).str[1:], errors="coerce")
        ctrl = set(map(str, ctrl_wells)) | set(map(str, extra_ctrl_wells))
        df["role"] = np.where(df["well"].astype(str).isin(ctrl), "ctr", "sample")
        if fit_df is not None:
            fit_cols = [c for c in ["well", "K", "sK"] if c in fit_df.columns]
            if "well" not in fit_cols and fit_df.index.name == "well":
                fit_df = fit_df.reset_index()
                fit_cols = [c for c in ["well", "K", "sK"] if c in fit_df.columns]
            if "well" in fit_cols:
                df = df.drop(columns=[c for c in fit_cols if c != "well" and c in df])
                df = df.merge(
                    fit_df[fit_cols], on="well", how="left", validate="many_to_one"
                )
        return type(self)(df, value_col=self.value_col)

    def step_centered(self, *, column: str | None = None) -> ResidualDiagnostics:
        """Return diagnostics with residuals centered within label and step."""
        column = column or self.value_col
        df = self.residuals.copy()
        centered_col = f"{column}_step_centered"
        df[centered_col] = df[column] - df.groupby(["label", "step"], observed=True)[
            column
        ].transform("mean")
        return type(self)(df, value_col=centered_col)

    def label_scaled(self, *, column: str | None = None) -> ResidualDiagnostics:
        """Return diagnostics with residuals divided by label-wise standard deviation."""
        column = column or self.value_col
        df = self.residuals.copy()
        scaled_col = f"{column}_label_scaled"
        scale = df.groupby("label", observed=True)[column].transform("std")
        df[scaled_col] = df[column] / scale.replace(0.0, np.nan)
        return type(self)(df, value_col=scaled_col)

    def with_relative_residuals(
        self,
        *,
        denominator: str = "yhat",
        raw_col: str = "raw_res",
        eps: float = 1e-12,
    ) -> ResidualDiagnostics:
        """Return diagnostics with raw and relative residual columns added."""
        df = self.residuals.copy()
        if raw_col not in df:
            df[raw_col] = df["y"] - df["yhat"]
        denom = df[denominator].abs().clip(lower=eps)
        df["rel_res"] = df[raw_col] / denom
        df["abs_rel_res"] = df["rel_res"].abs()
        return type(self)(df, value_col=self.value_col)

    def well_scaled(
        self,
        *,
        column: str | None = None,
        min_count: int = 3,
    ) -> ResidualDiagnostics:
        """Return diagnostics with residuals divided by well-and-label SD."""
        column = column or self.value_col
        df = self.residuals.copy()
        group = df.groupby(["well", "label"], observed=True)[column]
        scale = group.transform("std")
        count = group.transform("count")
        scale_col = f"{column}_well_sd"
        scaled_col = f"{column}_well_scaled"
        df[scale_col] = scale.where(count >= min_count)
        df[scaled_col] = df[column] / df[scale_col].replace(0.0, np.nan)
        return type(self)(df, value_col=scaled_col)

    def well_summary(self, *, column: str | None = None) -> pd.DataFrame:
        """Summarize residual and signal behavior per well and label."""
        column = column or self.value_col
        df = self.residuals.copy()
        if "raw_res" not in df and {"y", "yhat"}.issubset(df.columns):
            df["raw_res"] = df["y"] - df["yhat"]
        if "rel_res" not in df and {"raw_res", "yhat"}.issubset(df.columns):
            denom = df["yhat"].abs().clip(lower=1e-12)
            df["rel_res"] = df["raw_res"] / denom

        agg: dict[str, pd.NamedAgg] = {
            "n": pd.NamedAgg(column=column, aggfunc="count"),
            "std_res_mean": pd.NamedAgg(column=column, aggfunc="mean"),
            "std_res_sd": pd.NamedAgg(column=column, aggfunc="std"),
            "y_mean": pd.NamedAgg(column="y", aggfunc="mean"),
            "yhat_mean": pd.NamedAgg(column="yhat", aggfunc="mean"),
            "sigma_med": pd.NamedAgg(column="sigma", aggfunc="median"),
        }
        if "raw_res" in df:
            agg["raw_sd"] = pd.NamedAgg(column="raw_res", aggfunc="std")
        if "rel_res" in df:
            agg["rel_sd"] = pd.NamedAgg(column="rel_res", aggfunc="std")
        if "K" in df:
            agg["K"] = pd.NamedAgg(column="K", aggfunc="first")
        if "role" in df:
            agg["role"] = pd.NamedAgg(column="role", aggfunc="first")
        if "row" in df:
            agg["row"] = pd.NamedAgg(column="row", aggfunc="first")
        if "col" in df:
            agg["col"] = pd.NamedAgg(column="col", aggfunc="first")

        return df.groupby(["well", "label"], observed=True).agg(**agg).reset_index()

    def normality(self, *, column: str | None = None) -> pd.DataFrame:
        """Return Shapiro, D'Agostino, and Anderson normality diagnostics."""
        column = column or self.value_col
        rows = [
            {"group": f"label_{label}", **_normality_row(group[column])}
            for label, group in self.residuals.groupby("label", observed=True)
        ]
        rows.append({"group": "pooled", **_normality_row(self.residuals[column])})
        return pd.DataFrame(rows)

    def step_summary(self, *, column: str | None = None) -> pd.DataFrame:
        """Summarize residuals by label and titration step."""
        column = column or self.value_col
        return self.residuals.groupby(["label", "step"], observed=True)[column].agg([
            "count",
            "mean",
            "std",
            "median",
        ])

    def position_summary(self, *, column: str | None = None) -> dict[str, pd.DataFrame]:
        """Summarize residuals by row, column, edge column, and role when present."""
        column = column or self.value_col
        df = self.residuals
        out: dict[str, pd.DataFrame] = {}
        if "row" in df:
            out["row"] = df.groupby(["label", "row"], observed=True)[column].agg([
                "count",
                "mean",
                "std",
                "skew",
            ])
        if "col" in df:
            out["col"] = df.groupby(["label", "col"], observed=True)[column].agg([
                "count",
                "mean",
                "std",
                "skew",
            ])
            edge = df.assign(edge_col=df["col"].isin([1, 12]))
            out["edge_col"] = edge.groupby(["label", "edge_col"], observed=True)[
                column
            ].agg(["count", "mean", "std", "skew"])
        if "role" in df:
            out["role"] = df.groupby(["label", "role"], observed=True)[column].agg([
                "count",
                "mean",
                "std",
                "skew",
            ])
        return out

    @property
    def analysis(self) -> ResidualAnalysis:
        """Structural analyses (covariance, correlation, bias, QA) over the table.

        Returns
        -------
        ResidualAnalysis
            Accessor exposing :meth:`~ResidualAnalysis.covariance`,
            :meth:`~ResidualAnalysis.correlation`,
            :meth:`~ResidualAnalysis.label_bias`, and
            :meth:`~ResidualAnalysis.validate`.
        """
        return ResidualAnalysis(self.residuals)

    def tail_rows(self, n: int = 30, *, column: str | None = None) -> pd.DataFrame:
        """Return rows with largest absolute residual values."""
        column = column or self.value_col
        cols = [
            "well",
            "label",
            "step",
            "x",
            "y",
            "yhat",
            "sigma",
            "std_res",
            column,
        ]
        cols = list(dict.fromkeys(c for c in cols if c in self.residuals.columns))
        return (
            self.residuals
            .assign(abs_res=self.residuals[column].abs())
            .sort_values("abs_res", ascending=False)
            .loc[:, cols]
            .head(n)
        )

    def plot_hist_qq(self, *, column: str | None = None) -> _t.Any:
        """Plot histogram and Q-Q panels by label plus pooled."""
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import seaborn as sns  # type: ignore[import-untyped]  # noqa: PLC0415

        column = column or self.value_col
        groups = [
            (f"label {label}", group[column].dropna())
            for label, group in self.residuals.groupby("label", observed=True)
        ]
        groups.append(("pooled", self.residuals[column].dropna()))
        fig, axes = plt.subplots(len(groups), 2, figsize=(10, 4 * len(groups)))
        axes_arr = np.asarray(axes).reshape(len(groups), 2)
        for row, (name, values) in enumerate(groups):
            sns.histplot(values, kde=True, ax=axes_arr[row, 0])
            axes_arr[row, 0].axvline(0, color="black", lw=1)
            axes_arr[row, 0].set_title(f"{name}: residual histogram")
            sp_stats.probplot(values, dist="norm", plot=axes_arr[row, 1])
            axes_arr[row, 1].set_title(f"{name}: Q-Q plot")
        fig.tight_layout()
        return fig

    def plot_step(self, *, column: str | None = None) -> _t.Any:
        """Plot residual distributions by label and titration step."""
        import seaborn as sns  # noqa: PLC0415

        column = column or self.value_col
        return sns.catplot(
            data=self.residuals,
            x="step",
            y=column,
            col="label",
            kind="box",
            sharey=True,
        )

    def plot_role(self, *, column: str | None = None) -> _t.Any:
        """Plot residual distributions by role, if diagnostics are annotated."""
        import seaborn as sns  # noqa: PLC0415

        if "role" not in self.residuals:
            msg = "Call annotate(ctrl_wells=...) before plot_role()."
            raise ValueError(msg)
        column = column or self.value_col
        return sns.catplot(
            data=self.residuals,
            x="role",
            y=column,
            col="label",
            kind="box",
            sharey=True,
        )

    def plot_col(self, *, column: str | None = None) -> _t.Any:
        """Plot residual distributions by plate column."""
        import seaborn as sns  # noqa: PLC0415

        if "col" not in self.residuals:
            msg = "Call annotate(...) before plot_col()."
            raise ValueError(msg)
        column = column or self.value_col
        return sns.catplot(
            data=self.residuals,
            x="col",
            y=column,
            col="label",
            kind="box",
            sharey=True,
        )

    def plot_well_summary(
        self,
        *,
        x: str = "yhat_mean",
        y: str = "std_res_sd",
        hue: str = "role",
    ) -> _t.Any:
        """Plot per-well residual spread against signal or fitted parameters."""
        import seaborn as sns  # noqa: PLC0415

        summary = self.well_summary()
        kwargs: dict[str, _t.Any] = {
            "data": summary,
            "x": x,
            "y": y,
            "col": "label",
            "kind": "scatter",
            "facet_kws": {"sharex": False, "sharey": False},
        }
        if hue in summary:
            kwargs["hue"] = hue
        return sns.relplot(**kwargs)


@dataclass
class ResidualComparison:
    """Bundle residual diagnostics, well summaries, and trace parameter summaries."""

    diagnostics: ResidualDiagnostics
    well: pd.DataFrame
    parameters: pd.DataFrame

    @classmethod
    def from_fit_results(  # noqa: PLR0913
        cls,
        results: dict[str, _t.Any],
        trace_id: str,
        binding_function: _t.Callable[..., ArrayLike],
        *,
        fit_df: pd.DataFrame | None = None,
        ctrl_wells: _t.Iterable[str] = (),
        extra_ctrl_wells: _t.Iterable[str] = (),
        robust: bool = False,
        value_col: str = "std_res",
    ) -> ResidualComparison:
        """Create a compact residual-comparison object from fit results."""
        diagnostics = (
            ResidualDiagnostics
            .from_fit_results(
                results,
                trace_id,
                binding_function,
                robust=robust,
                value_col=value_col,
            )
            .annotate(
                fit_df=fit_df,
                ctrl_wells=ctrl_wells,
                extra_ctrl_wells=extra_ctrl_wells,
            )
            .with_relative_residuals()
        )
        parameters = trace_parameter_summary(results)
        well = diagnostics.well_summary()
        if not parameters.empty:
            well = well.merge(parameters, on=["well", "label"], how="left")
        return cls(diagnostics=diagnostics, well=well, parameters=parameters)

    def with_value(self, column: str) -> ResidualComparison:
        """Return a copy using a different residual column as the active value."""
        diagnostics = type(self.diagnostics)(
            self.diagnostics.residuals, value_col=column
        )
        return type(self)(
            diagnostics=diagnostics,
            well=diagnostics.well_summary(),
            parameters=self.parameters.copy(),
        )


def _normality_row(values: pd.Series) -> dict[str, float | int]:
    """Return normality-test statistics for one residual vector."""
    x = values.dropna().to_numpy(dtype=float)
    row: dict[str, float | int] = {"n": int(x.size)}
    if x.size >= 3:
        shapiro = sp_stats.shapiro(x)
        row["shapiro_W"] = float(shapiro.statistic)
        row["shapiro_p"] = float(shapiro.pvalue)
    if x.size >= 8:
        dag = sp_stats.normaltest(x)
        row["dagostino_K2"] = float(dag.statistic)
        row["dagostino_p"] = float(dag.pvalue)
    if x.size >= 5:
        row["anderson_A2"] = float(sp_stats.anderson(x, dist="norm").statistic)
    return row


def residual_normal_scores(
    likelihood_residual: ArrayLike,
    *,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Map likelihood-scale residuals onto a Normal diagnostic scale.

    For Normal likelihoods this is the identity.  For Student-t likelihoods,
    ``(y - mu) / sigma`` follows a t distribution, so Normal QQ plots and
    ``abs(residual) > 2`` style diagnostics should use the probability integral
    transform to an equivalent standard-Normal score.
    """
    r = np.asarray(likelihood_residual, dtype=float)
    if not robust:
        return r

    probs = sp_stats.t.cdf(r, df=student_t_nu)
    probs = np.clip(probs, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return np.asarray(sp_stats.norm.ppf(probs), dtype=float)


def robust_residual_outlier_mask(
    likelihood_residual: ArrayLike,
    *,
    threshold: float = 3.0,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Flag observations by calibrated Normal-score residual magnitude."""
    z = residual_normal_scores(
        likelihood_residual, robust=robust, student_t_nu=student_t_nu
    )
    return np.asarray(np.isfinite(z) & (np.abs(z) > threshold), dtype=bool)


def excess_tail_outlier_mask(  # noqa: PLR0913
    likelihood_residual: ArrayLike,
    *,
    threshold: float = 3.0,
    allowed_tail_fraction: float = 0.01,
    min_allowed_tail_count: int = 1,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
) -> np.ndarray:
    """Mask only residual outliers beyond an allowed tail fraction.

    The residuals are first mapped to the calibrated Normal diagnostic scale.
    Observations with ``abs(z) <= threshold`` are never removed.  If more than
    ``allowed_tail_fraction`` of finite observations exceed the threshold, only
    the largest excess observations are marked for removal.
    """
    z = residual_normal_scores(
        likelihood_residual, robust=robust, student_t_nu=student_t_nu
    )
    z = np.asarray(z, dtype=float)
    remove = np.zeros(z.shape, dtype=bool)
    finite = np.isfinite(z)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return remove

    candidate_idx = np.flatnonzero(finite & (np.abs(z) > threshold))
    allowed = max(
        int(min_allowed_tail_count), int(np.floor(allowed_tail_fraction * n_finite))
    )
    n_remove = max(0, int(candidate_idx.size) - allowed)
    if n_remove == 0:
        return remove

    order = np.argsort(np.abs(z[candidate_idx]))[::-1]
    remove[candidate_idx[order[:n_remove]]] = True
    return remove


def mark_excess_residual_outliers(  # noqa: PLR0913
    residuals: pd.DataFrame,
    *,
    residual_col: str = "std_res",
    group_cols: tuple[str, ...] = ("trace_id", "label"),
    threshold: float = 3.0,
    allowed_tail_fraction: float = 0.01,
    min_allowed_tail_count: int = 1,
    exclude_col: str = "exclude_residual_outlier",
) -> pd.DataFrame:
    """Annotate residual rows to remove only excess calibrated tail outliers."""
    out = residuals.copy()
    out[exclude_col] = False
    out["residual_outlier_score"] = np.nan

    actual_group_cols = [col for col in group_cols if col in out.columns]
    grouped: _t.Iterable[tuple[object, pd.DataFrame]]
    if actual_group_cols:
        grouped = out.groupby(actual_group_cols, observed=True, sort=False)
    else:
        grouped = [(None, out)]

    for _key, group in grouped:
        values = group[residual_col].to_numpy(dtype=float)
        remove = excess_tail_outlier_mask(
            values,
            threshold=threshold,
            allowed_tail_fraction=allowed_tail_fraction,
            min_allowed_tail_count=min_allowed_tail_count,
        )
        out.loc[group.index, "residual_outlier_score"] = np.abs(values)
        out.loc[group.index[remove], exclude_col] = True

    return out


def masked_datasets_from_residual_outliers(
    results: _t.Mapping[str, _t.Any],
    residuals: pd.DataFrame,
    *,
    exclude_col: str = "exclude_residual_outlier",
    min_keep: int = 3,
) -> dict[str, _t.Any]:
    """Return datasets with residual rows marked by *exclude_col* masked out.

    This is intended for the second pass of a sensitivity analysis: fit once,
    compute residuals, annotate excess-tail outliers, mask those rows, then refit.
    """
    masked: dict[str, _t.Any] = {}
    for well, fr in results.items():
        dataset = getattr(fr, "dataset", None)
        if dataset is not None:
            masked[str(well)] = copy.deepcopy(dataset)
        elif hasattr(fr, "items"):
            masked[str(well)] = copy.deepcopy(fr)

    if not masked or exclude_col not in residuals.columns:
        return masked

    drop_rows = residuals[residuals[exclude_col].astype(bool)].copy()
    if drop_rows.empty:
        return masked

    if "residual_outlier_score" not in drop_rows.columns:
        drop_rows["residual_outlier_score"] = np.abs(
            drop_rows.get("std_res", pd.Series(np.nan, index=drop_rows.index))
        )
    drop_rows = drop_rows.sort_values("residual_outlier_score", ascending=False)

    for row in drop_rows.itertuples(index=False):
        well = str(row.well)
        label = str(row.label)
        if well not in masked or label not in masked[well]:
            continue

        da = masked[well][label]
        if int(np.sum(da.mask)) <= min_keep:
            continue

        raw_i = getattr(row, "raw_i", None)
        if raw_i is None or pd.isna(raw_i):
            raw_i = getattr(row, "step", None)
        if raw_i is None or pd.isna(raw_i):
            continue

        idx = int(raw_i)
        if 0 <= idx < len(da.mask):
            da.mask[idx] = False

    return masked


def mark_outlier_probability_outliers(
    residuals: _t.Any,
    *,
    probability_col: str = "p_outlier_per_point",
    threshold: float = 0.9,
    exclude_col: str = "exclude_outlier_probability",
) -> pd.DataFrame:
    """Annotate rows with high posterior outlier probability.

    Parameters
    ----------
    residuals : _t.Any
        Residual table, usually returned by :func:`residuals_from_multifit` or
        :func:`residuals_from_fit_results`.
    probability_col : str, optional
        Column containing per-point posterior outlier probabilities.
    threshold : float, optional
        Probability cutoff above which a row is marked as an outlier.
    exclude_col : str, optional
        Boolean output column used to mark rows for exclusion.

    Returns
    -------
    pd.DataFrame
        Copy of ``residuals`` with ``exclude_col`` and
        ``residual_outlier_score`` columns added.

    Raises
    ------
    TypeError
        If ``residuals`` is not a pandas DataFrame.
    """
    if not isinstance(residuals, pd.DataFrame):
        msg = (
            "residuals must be a pandas DataFrame, such as the output of "
            "residuals_from_multifit() or residuals_from_fit_results(); got "
            f"{type(residuals).__name__}"
        )
        raise TypeError(msg)
    out = residuals.copy()
    probability_values = (
        out[probability_col]
        if probability_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    probabilities = pd.to_numeric(probability_values, errors="coerce")
    out[exclude_col] = probabilities > threshold
    out["residual_outlier_score"] = probabilities
    return out


def masked_datasets_from_outlier_probabilities(
    results: _t.Mapping[str, _t.Any],
    residuals: _t.Any,
    *,
    probability_col: str = "p_outlier_per_point",
    threshold: float = 0.9,
    min_keep: int = 3,
) -> dict[str, _t.Any]:
    """Mask datasets using posterior outlier probabilities.

    Parameters
    ----------
    results : _t.Mapping[str, _t.Any]
        Mapping from well identifiers to datasets or fit-result-like objects
        containing datasets.
    residuals : _t.Any
        Residual table with pointwise posterior outlier probabilities.
    probability_col : str, optional
        Column containing per-point posterior outlier probabilities.
    threshold : float, optional
        Probability cutoff above which a row is masked.
    min_keep : int, optional
        Minimum number of unmasked points retained per label.

    Returns
    -------
    dict[str, _t.Any]
        Deep-copied datasets with high-probability outlier rows masked.

    Raises
    ------
    TypeError
        If ``residuals`` is not a pandas DataFrame.
    """
    if not isinstance(residuals, pd.DataFrame):
        msg = (
            "residuals must be a pandas DataFrame, such as the output of "
            "residuals_from_multifit() or residuals_from_fit_results(); got "
            f"{type(residuals).__name__}"
        )
        raise TypeError(msg)
    exclude_col = "exclude_outlier_probability"
    marked = mark_outlier_probability_outliers(
        residuals,
        probability_col=probability_col,
        threshold=threshold,
        exclude_col=exclude_col,
    )
    return masked_datasets_from_residual_outliers(
        results,
        marked,
        exclude_col=exclude_col,
        min_keep=min_keep,
    )


def posterior_dataset(trace: _t.Any) -> _t.Any:
    """Return the posterior xarray Dataset from ArviZ InferenceData or DataTree.

    PyMC/ArviZ versions differ in whether returned objects are InferenceData-like
    or xarray DataTree-like.  This helper hides that difference for validation code.
    """
    if hasattr(trace, "posterior"):
        return trace.posterior
    node = trace["posterior"]
    return getattr(node, "ds", node)


def sample_stats_dataset(trace: _t.Any) -> _t.Any:
    """Return sample_stats Dataset from InferenceData or DataTree."""
    if hasattr(trace, "sample_stats"):
        return trace.sample_stats
    node = trace["sample_stats"]
    return getattr(node, "ds", node)


def trace_parameter_summary(results: _t.Mapping[str, _t.Any]) -> pd.DataFrame:
    """Summarize scalar per-label PyMC noise parameters from traces.

    Parameters
    ----------
    results : _t.Mapping[str, _t.Any]
        Mapping from well identifiers to fit-result-like objects with ``mini``
        or ``trace`` attributes and optional datasets.

    Returns
    -------
    pd.DataFrame
        Wide table indexed by well and label with posterior means and standard
        deviations for recognized scalar noise parameters. An empty table with
        ``well`` and ``label`` columns is returned when no recognized trace
        parameters are available.
    """
    rows: list[dict[str, _t.Any]] = []
    for well, fit in results.items():
        trace = getattr(fit, "mini", None)
        if trace is None:
            trace = getattr(fit, "trace", None)
        if trace is None:
            continue
        posterior = _posterior_dataset_or_none(trace)
        if posterior is None:
            continue

        labels = _fit_result_labels(fit)
        for var_name in posterior.data_vars:
            parsed = _parse_trace_parameter_name(str(var_name))
            if parsed is None:
                continue
            parameter, label = parsed
            target_labels = labels if label == "shared" and labels else (label,)
            values_da = posterior[var_name]
            if "well" in values_da.dims:
                try:
                    values_da = values_da.sel(well=str(well))
                except KeyError:
                    continue
            values = values_da.values
            sd = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
            rows.extend(
                {
                    "well": str(well),
                    "label": str(target_label),
                    "parameter": parameter,
                    "mean": float(np.nanmean(values)),
                    "sd": sd,
                }
                for target_label in target_labels
            )

    if not rows:
        return pd.DataFrame(columns=["well", "label"])

    long = pd.DataFrame(rows)
    wide = long.pivot_table(
        index=["well", "label"],
        columns="parameter",
        values=["mean", "sd"],
        aggfunc="first",
    )
    flat_columns = np.asarray(wide.columns, dtype=object)
    names = []
    for column in flat_columns:
        if isinstance(column, tuple) and len(column) == 2:
            stat, param = column
        else:
            stat, param = "value", column
        names.append(f"{param!s}_{stat!s}")
    wide.columns = names
    return wide.reset_index()


def _posterior_dataset_or_none(trace: _t.Any) -> _t.Any | None:
    try:
        return posterior_dataset(trace)
    except Exception:
        return None


def robust_settings_from_trace(trace: _t.Any) -> tuple[bool, float]:
    """Infer ``(robust, student_t_nu)`` from a trace's posterior variables.

    Detects an inferred Student-t (a ``student_t_nu`` deterministic) or a
    contamination mixture (``pi_outlier_*`` / ``outlier_inflate``). Used by
    ``FitResult.residuals`` / ``MultiFitResult.residuals`` so the residual table
    is standardized correctly without the caller re-supplying fit settings.

    Parameters
    ----------
    trace : _t.Any
        A PyMC trace (``xr.DataTree``) or ``None`` for classical fits.

    Returns
    -------
    tuple[bool, float]
        Whether a robust likelihood was used and the Student-t nu to apply.

    Notes
    -----
    A *fixed*-nu Student-t leaves no trace marker and is reported as non-robust;
    only the Normal-score transform of ``std_res`` differs (``raw_res``/``yhat``/
    ``likelihood_res`` are unaffected). Pass ``robust=`` to
    ``residual_table`` to override.
    """
    post = _posterior_dataset_or_none(trace)
    if post is None:
        return False, STUDENT_T_NU
    names = {str(n) for n in getattr(post, "data_vars", {})}
    if any(n.startswith("pi_outlier") for n in names) or "outlier_inflate" in names:
        return True, STUDENT_T_NU
    if "student_t_nu" in names:
        try:
            return True, float(post["student_t_nu"].mean())
        except Exception:
            return True, STUDENT_T_NU
    return False, STUDENT_T_NU


def _fit_result_labels(fit: _t.Any) -> tuple[str, ...]:
    dataset = getattr(fit, "dataset", None)
    if dataset is None:
        return ()
    try:
        return tuple(str(label) for label in dataset)
    except TypeError:
        return ()


def _parse_trace_parameter_name(var_name: str) -> tuple[str, str] | None:
    prefixes = {
        "label_noise_scale": "label_noise_scale",
        "well_noise_scale": "well_noise_scale",
        "well_noise_sd": "well_noise_sd",
        "ye_mag": "ye_mag",
        "rel_error": "rel_error",
        "floor": "floor",
        "gain": "gain",
    }
    for prefix, parameter in prefixes.items():
        if var_name == prefix:
            return parameter, "shared"
        if var_name.startswith(f"{prefix}_"):
            return parameter, var_name.removeprefix(f"{prefix}_")
    return None


def _posterior_mean_scalar(
    posterior: _t.Any, var_name: str, default: float = 1.0
) -> float:
    if var_name not in posterior:
        return default
    return float(posterior[var_name].mean(("chain", "draw")).values)


def x_axis_sanity(trace: _t.Any) -> dict[str, _t.Any]:
    """Check pH/x-axis invariants for traces with ``x_per_well``.

    For per-well x models with a shared start pH, all wells at step 0 should be
    identical within each draw.  ``x_step0_max_abs_spread`` should therefore be
    close to zero.
    """
    out: dict[str, _t.Any] = {}
    try:
        posterior = posterior_dataset(trace)
        if "x_per_well" not in posterior:
            return out
        x = posterior["x_per_well"]
        if "step" not in x.dims or "well" not in x.dims:
            out["x_sanity_error"] = (
                f"x_per_well dims are {x.dims!r}, expected step/well"
            )
            return out
        x0 = x.isel(step=0)
        x0_ref = x0.isel(well=0)
        out["x_step0_max_abs_spread"] = float(np.abs(x0 - x0_ref).max().values)
        if "well" in x.coords:
            first_well = str(x.coords["well"].values[0])
            traj = x.sel(well=first_well).mean(("chain", "draw")).values
            out["x_first_well"] = first_well
            out["x_first_well_step0_mean"] = float(traj[0])
            out["x_first_well_last_mean"] = float(traj[-1])
    except Exception as e:
        out["x_sanity_error"] = repr(e)
    return out


def trace_diagnostics(
    trace: _t.Any,
    *,
    compute_loo: bool = False,
    summary_var_names: list[str] | None = None,
) -> dict[str, _t.Any]:
    """Collect basic MCMC and optional LOO diagnostics from a PyMC trace."""
    row: dict[str, _t.Any] = {}

    try:
        ss = sample_stats_dataset(trace)
        if "diverging" in ss:
            row["n_divergences"] = int(ss["diverging"].sum().values)
        if "tree_depth" in ss:
            row["tree_depth_max"] = int(ss["tree_depth"].max().values)
        if "reached_max_treedepth" in ss:
            row["n_reached_max_treedepth"] = int(
                ss["reached_max_treedepth"].sum().values
            )
        if "energy" in ss:
            energy = ss["energy"].values.ravel()
            row["energy_sd"] = float(np.nanstd(energy))
    except Exception as e:
        row["sample_stats_error"] = repr(e)

    if summary_var_names is None:
        summary_var_names = [
            "K",
            "K_ctr",
            "x_start",
            "x_true",
            "x_per_well",
            "acid_drop_global",
            "acid_drop_well",
            "x_step",
            "ye_mag",
            "rel_error",
            "floor",
            "gain",
        ]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = az.summary(trace, var_names=summary_var_names, filter_vars="like")
        if "r_hat" in summary:
            row["rhat_max"] = float(summary["r_hat"].max(skipna=True))
        if "ess_bulk" in summary:
            row["ess_bulk_min"] = float(summary["ess_bulk"].min(skipna=True))
        if "ess_tail" in summary:
            row["ess_tail_min"] = float(summary["ess_tail"].min(skipna=True))
        _add_warnings(row, "summary", caught)
    except Exception as e:
        row["summary_error"] = repr(e)

    if compute_loo:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loo = az.loo(merge_log_likelihoods(trace), var_name="obs")
            row["elpd_loo"] = _loo_value(loo, "elpd_loo", "elpd")
            row["p_loo"] = _loo_value(loo, "p_loo", "p")
            row["loo_se"] = _loo_value(loo, "se", "elpd_loo_se", "elpd_se")
            if hasattr(loo, "pareto_k"):
                pk = np.asarray(loo.pareto_k).ravel()
                row["pareto_k_max"] = float(np.nanmax(pk))
                row["pareto_k_frac_gt_0p7"] = float(np.nanmean(pk > 0.7))
            _add_warnings(row, "loo", caught)
        except Exception as e:
            row["loo_error"] = repr(e)

    row.update(x_axis_sanity(trace))
    return row


def pareto_k_table(
    multi_or_trace: _t.Any, results: _t.Mapping[str, _t.Any] | None = None
) -> pd.DataFrame:
    """Return pointwise PSIS-LOO Pareto-k values annotated by well, label, and step.

    Parameters
    ----------
    multi_or_trace : _t.Any
        A ``MultiFitResult``-like object with ``trace`` and ``results`` attributes,
        or a raw PyMC/ArviZ trace.
    results : _t.Mapping[str, _t.Any] | None
        Fit results keyed by well. Required when *multi_or_trace* is a raw trace.

    Returns
    -------
    pd.DataFrame
        One row per likelihood observation with ``pareto_k`` and observation
        metadata where it can be recovered from the fitted datasets.

    Raises
    ------
    ValueError
        If no fit-result mapping is available, if the trace lacks pointwise
        log-likelihood data, or if ArviZ's pointwise output cannot be aligned
        to the fitted datasets.
    """
    trace = getattr(multi_or_trace, "trace", multi_or_trace)
    fit_results = (
        results if results is not None else getattr(multi_or_trace, "results", None)
    )
    if fit_results is None:
        msg = "Pass a MultiFitResult or provide results= when using a raw trace."
        raise ValueError(msg)
    if not hasattr(trace, "log_likelihood"):
        msg = (
            "Trace does not contain a log_likelihood group; "
            "run pointwise log-likelihood first."
        )
        raise ValueError(msg)

    rows: list[dict[str, _t.Any]] = []
    for var_name in trace.log_likelihood.data_vars:
        obs_rows = _observation_rows_for_likelihood(str(var_name), fit_results)
        loo = az.loo(trace, var_name=str(var_name), pointwise=True)
        pareto_k = np.asarray(loo.pareto_k, dtype=float).ravel()
        if len(obs_rows) != pareto_k.size:
            msg = (
                f"Pareto-k length mismatch for {var_name!r}: "
                f"{pareto_k.size} values but {len(obs_rows)} mapped observations."
            )
            raise ValueError(msg)
        for obs_idx, (row, k_value) in enumerate(zip(obs_rows, pareto_k, strict=True)):
            rows.append({
                "log_likelihood": str(var_name),
                "obs_index": obs_idx,
                **row,
                "pareto_k": float(k_value),
                "pareto_k_warn": bool(np.isfinite(k_value) and k_value > 0.7),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "log_likelihood",
                "obs_index",
                "well",
                "label",
                "step",
                "x",
                "y",
                "pareto_k",
                "pareto_k_warn",
            ]
        )
    return pd.DataFrame(rows)


def pareto_k_summary(pareto_k: pd.DataFrame) -> pd.DataFrame:
    """Summarize pointwise Pareto-k diagnostics.

    Parameters
    ----------
    pareto_k : pd.DataFrame
        Pointwise Pareto-k table, usually returned by
        :func:`pointwise_pareto_k`.

    Returns
    -------
    pd.DataFrame
        Summary table grouped by label and well when those columns are present,
        otherwise grouped by likelihood variable. The table includes count,
        maximum, mean, and warning fraction. Empty input returns an empty table.
    """
    if pareto_k.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("label", "well") if col in pareto_k.columns]
    if not group_cols:
        group_cols = ["log_likelihood"]
    return (
        pareto_k
        .groupby(group_cols, observed=True)
        .agg(
            n=("pareto_k", "count"),
            pareto_k_max=("pareto_k", "max"),
            pareto_k_mean=("pareto_k", "mean"),
            pareto_k_frac_gt_0p7=("pareto_k_warn", "mean"),
        )
        .reset_index()
    )


def _observation_rows_for_likelihood(
    var_name: str,
    results: _t.Mapping[str, _t.Any],
) -> list[dict[str, _t.Any]]:
    """Map one likelihood variable's vectorized observations to dataset rows."""
    label = _label_from_likelihood_name(var_name)
    if label is None:
        return _generic_observation_rows(var_name, results)

    rows: list[dict[str, _t.Any]] = []
    active: list[tuple[str, _t.Any, _t.Any]] = []
    for well, fit in results.items():
        ds = getattr(fit, "dataset", None)
        if ds is not None and label in ds:
            active.append((str(well), ds, ds[label]))
    if not active:
        return rows

    n_steps = len(np.asarray(active[0][2].mask, dtype=bool))
    for step in range(n_steps):
        for well, ds, da in active:
            mask = np.asarray(da.mask, dtype=bool)
            if step >= mask.size or not bool(mask[step]):
                continue
            row: dict[str, _t.Any] = {
                "well": well,
                "label": str(label),
                "step": int(step),
            }
            if hasattr(da, "xc") and np.asarray(da.xc).size == mask.size:
                row["x"] = float(np.asarray(da.xc, dtype=float)[step])
            elif hasattr(da, "x") and np.asarray(da.x).size:
                compact_idx = int(mask[: step + 1].sum() - 1)
                row["x"] = float(np.asarray(da.x, dtype=float)[compact_idx])
            if hasattr(da, "yc") and np.asarray(da.yc).size == mask.size:
                row["y"] = float(np.asarray(da.yc, dtype=float)[step])
            elif hasattr(da, "y") and np.asarray(da.y).size:
                compact_idx = int(mask[: step + 1].sum() - 1)
                row["y"] = float(np.asarray(da.y, dtype=float)[compact_idx])
            row["is_ph"] = bool(getattr(ds, "is_ph", False))
            rows.append(row)
    return rows


def _label_from_likelihood_name(var_name: str) -> str | None:
    prefix = "y_likelihood_"
    if not var_name.startswith(prefix):
        return None
    return var_name.removeprefix(prefix)


def _generic_observation_rows(
    var_name: str,
    results: _t.Mapping[str, _t.Any],
) -> list[dict[str, _t.Any]]:
    """Fallback mapping for nonstandard likelihood variable names."""
    rows: list[dict[str, _t.Any]] = []
    for well, fit in results.items():
        ds = getattr(fit, "dataset", None)
        if ds is None:
            continue
        for label, da in ds.items():
            mask = np.asarray(da.mask, dtype=bool)
            rows.extend(
                {
                    "well": str(well),
                    "label": str(label),
                    "step": int(step),
                    "log_likelihood_hint": var_name,
                }
                for step in np.flatnonzero(mask)
            )
    return rows


def _add_warnings(
    row: dict[str, _t.Any], prefix: str, caught: list[warnings.WarningMessage]
) -> None:
    if not caught:
        return
    messages = [str(w.message) for w in caught]
    row[f"{prefix}_warning_count"] = len(messages)
    row[f"{prefix}_warnings"] = " | ".join(dict.fromkeys(messages))


def _loo_value(loo: _t.Any, *names: str) -> float:
    lookup_errors: list[str] = []
    for name in names:
        if hasattr(loo, name):
            return float(getattr(loo, name))
        try:
            return float(loo[name])
        except Exception as e:
            lookup_errors.append(f"{name}: {e!r}")
    msg = (
        f"LOO result has none of {names!r}; available fields: {dir(loo)!r}; "
        f"lookup errors: {lookup_errors!r}"
    )
    raise AttributeError(msg)


def merge_log_likelihoods(trace: _t.Any) -> _t.Any:
    """Merge multiple pointwise log-likelihood variables for ArviZ LOO/compare."""
    if not hasattr(trace, "log_likelihood"):
        return trace
    ll = trace.log_likelihood
    data_vars = getattr(ll, "data_vars", {})
    if len(data_vars) <= 1 and "obs" in data_vars:
        return trace

    data_list = []
    current_idx = 0
    for var_name in data_vars:
        data = ll[var_name]
        last_dim = data.dims[-1]
        n_obs = data.sizes[last_dim]
        data = data.rename({last_dim: "obs_id"})
        data = data.assign_coords(obs_id=np.arange(current_idx, current_idx + n_obs))
        data_list.append(data)
        current_idx += n_obs
    if not data_list:
        return trace

    groups = {
        "posterior": posterior_dataset(trace),
        "log_likelihood": xr.Dataset({"obs": xr.concat(data_list, dim="obs_id")}),
    }
    if hasattr(trace, "observed_data"):
        groups["observed_data"] = trace.observed_data
    return xr.DataTree.from_dict(groups)


def _sigma_for_label_well(
    trace: _t.Any, lbl: _t.Any, well: str, da: _t.Any, mask: np.ndarray
) -> np.ndarray:
    posterior = posterior_dataset(trace)

    sigma_var = f"sigma_obs_{lbl}"
    if sigma_var in posterior:
        arr = posterior[sigma_var]
        try:
            if "well" in arr.dims:
                sigma_full = arr.sel(well=well).mean(("chain", "draw")).values
            else:
                sigma_full = arr.mean(("chain", "draw")).values
        except Exception:
            sigma_full = arr.mean(("chain", "draw")).values
        sigma = np.asarray(sigma_full, dtype=float)
        if sigma.shape == mask.shape:
            sigma = sigma[mask]
        return np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)

    if hasattr(da, "y_err") and np.asarray(da.y_err).size == int(mask.sum()):
        sigma = np.asarray(da.y_err, dtype=float)
    elif hasattr(da, "y_errc") and np.asarray(da.y_errc).size == mask.size:
        sigma = np.asarray(da.y_errc, dtype=float)[mask]
    else:
        ye_var = f"ye_mag_{lbl}"
        ye_mag = _posterior_mean_scalar(posterior, ye_var, default=1.0)
        sigma = ye_mag * np.ones(int(mask.sum()), dtype=float)
    return np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)


def _outlier_probability_for_label_well(
    trace: _t.Any,
    lbl: _t.Any,
    well: str,
    mask: np.ndarray,
    results: _t.Mapping[str, _t.Any],
) -> np.ndarray:
    posterior = posterior_dataset(trace)
    prob_var = f"outlier_probability_{lbl}"
    n_active = int(mask.sum())
    missing = np.full(n_active, np.nan, dtype=float)
    if prob_var not in posterior:
        return missing

    arr = posterior[prob_var]
    sample_dims = [dim for dim in ("chain", "draw") if dim in arr.dims]
    if sample_dims:
        arr = arr.mean(dim=sample_dims)

    if "well" in arr.dims:
        try:
            well_arr = arr.sel(well=well)
        except Exception:
            well_arr = arr
        probs = np.asarray(well_arr, dtype=float)
        if probs.shape == mask.shape:
            return _t.cast("np.ndarray", probs[mask])
        if probs.size == n_active:
            return _t.cast("np.ndarray", probs.ravel())

    probs = np.asarray(arr, dtype=float).ravel()
    if probs.size == n_active:
        return _t.cast("np.ndarray", probs)

    obs_rows = _observation_rows_for_likelihood(f"y_likelihood_{lbl}", results)
    if probs.size != len(obs_rows):
        return missing

    by_position: dict[tuple[str, int], float] = {}
    for row, prob in zip(obs_rows, probs, strict=True):
        step_value = row.get("step")
        if step_value is None:
            continue
        by_position[str(row.get("well")), int(step_value)] = float(prob)
    step = np.flatnonzero(mask)
    return np.asarray(
        [by_position.get((str(well), int(step_j)), np.nan) for step_j in step],
        dtype=float,
    )


def residuals_from_multifit(  # noqa: PLR0913
    multi: _t.Any,
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Build a long calibrated-residual table from a MultiFitResult.

    The returned table has one row per active observation and always includes
    trace, well, label, step, observed, predicted, sigma, raw residual,
    likelihood-scaled residual, Normal-score residual, likelihood family, and
    residual-outlier metadata columns.

    ``likelihood_res`` is always ``(observed - predicted) / sigma``.  For
    Student-t robust fits, ``std_res`` is the equivalent standard-Normal score
    from the t CDF, suitable for Normal QQ plots and z-style outlier flags.
    """
    rows: list[dict[str, _t.Any]] = []

    for well, fr in multi.results.items():
        if fr.dataset is None or fr.result is None:
            continue
        ds = fr.dataset
        pars = fr.result.params

        for lbl, da in ds.items():
            mask = np.asarray(da.mask, dtype=bool)
            step = np.flatnonzero(mask)

            if hasattr(da, "xc") and np.asarray(da.xc).size == mask.size:
                x = np.asarray(da.xc, dtype=float)[mask]
            else:
                x = np.asarray(da.x, dtype=float)

            if hasattr(da, "yc") and np.asarray(da.yc).size == mask.size:
                y = np.asarray(da.yc, dtype=float)[mask]
            else:
                y = np.asarray(da.y, dtype=float)

            yhat = binding_function(
                x,
                pars["K"].value,
                pars[f"S0_{lbl}"].value,
                pars[f"S1_{lbl}"].value,
                is_ph=ds.is_ph,
            )
            sigma = _sigma_for_label_well(multi.trace, lbl, str(well), da, mask)
            p_outlier = _outlier_probability_for_label_well(
                multi.trace, lbl, str(well), mask, multi.results
            )
            likelihood_res = (y - yhat) / sigma
            std_res = residual_normal_scores(
                likelihood_res, robust=robust, student_t_nu=student_t_nu
            )
            outlier = robust_residual_outlier_mask(
                likelihood_res,
                threshold=outlier_threshold,
                robust=robust,
                student_t_nu=student_t_nu,
            )

            for j in range(len(y)):
                row = {
                    "trace_id": trace_id,
                    "well": str(well),
                    "label": str(lbl),
                    "step": int(step[j]),
                    "raw_i": int(step[j]),
                    "x": float(x[j]),
                    "y": float(y[j]),
                    "yhat": float(yhat[j]),
                    "sigma": float(sigma[j]),
                    "raw_res": float(y[j] - yhat[j]),
                    "likelihood_res": float(likelihood_res[j]),
                    "std_res": float(std_res[j]),
                    "p_outlier_per_point": float(p_outlier[j]),
                    "residual_likelihood": "student_t" if robust else "normal",
                    "student_t_nu": float(student_t_nu) if robust else np.nan,
                    "is_residual_outlier": bool(outlier[j]),
                    "outlier_threshold": float(outlier_threshold),
                }
                if include_fit_params:
                    row["K"] = float(pars["K"].value)
                    row[f"S0_{lbl}"] = float(pars[f"S0_{lbl}"].value)
                    row[f"S1_{lbl}"] = float(pars[f"S1_{lbl}"].value)
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=RESIDUAL_TABLE_COLUMNS)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["std_res", "x", "step", "label"]
    )
    leading_cols = [col for col in RESIDUAL_TABLE_COLUMNS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in leading_cols]
    return df.loc[:, [*leading_cols, *extra_cols]]


def residuals_from_fit_results(  # noqa: PLR0913
    results: dict[str, _t.Any],
    trace_id: str,
    binding_function: _t.Callable[..., ArrayLike],
    *,
    include_fit_params: bool = False,
    robust: bool = False,
    student_t_nu: float = STUDENT_T_NU,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Build a long calibrated residual table from classical FitResult objects."""
    rows: list[dict[str, _t.Any]] = []
    for well, fr in results.items():
        if fr.dataset is None or fr.result is None:
            continue
        ds = fr.dataset
        pars = fr.result.params
        for lbl, da in ds.items():
            mask = np.asarray(da.mask, dtype=bool)
            step = np.flatnonzero(mask)
            x = np.asarray(da.x, dtype=float)
            y = np.asarray(da.y, dtype=float)
            yhat = binding_function(
                x,
                pars["K"].value,
                pars[f"S0_{lbl}"].value,
                pars[f"S1_{lbl}"].value,
                is_ph=ds.is_ph,
            )
            if hasattr(da, "y_err") and np.asarray(da.y_err).size == len(y):
                sigma = np.asarray(da.y_err, dtype=float)
            else:
                sigma = np.ones_like(y, dtype=float)
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
            likelihood_res = (y - yhat) / sigma
            std_res = residual_normal_scores(
                likelihood_res, robust=robust, student_t_nu=student_t_nu
            )
            outlier = robust_residual_outlier_mask(
                likelihood_res,
                threshold=outlier_threshold,
                robust=robust,
                student_t_nu=student_t_nu,
            )
            for j in range(len(y)):
                row = {
                    "trace_id": trace_id,
                    "well": str(well),
                    "label": str(lbl),
                    "step": int(step[j]) if j < len(step) else int(j),
                    "raw_i": int(step[j]) if j < len(step) else int(j),
                    "x": float(x[j]),
                    "y": float(y[j]),
                    "yhat": float(yhat[j]),
                    "sigma": float(sigma[j]),
                    "raw_res": float(y[j] - yhat[j]),
                    "likelihood_res": float(likelihood_res[j]),
                    "std_res": float(std_res[j]),
                    "residual_likelihood": "student_t" if robust else "normal",
                    "student_t_nu": float(student_t_nu) if robust else np.nan,
                    "is_residual_outlier": bool(outlier[j]),
                    "outlier_threshold": float(outlier_threshold),
                }
                if include_fit_params:
                    row["K"] = float(pars["K"].value)
                    row[f"S0_{lbl}"] = float(pars[f"S0_{lbl}"].value)
                    row[f"S1_{lbl}"] = float(pars[f"S1_{lbl}"].value)
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["std_res", "x", "step", "label"]
    )


def mad(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    med = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - med)))


def residual_distribution_summary(res_df: pd.DataFrame) -> pd.DataFrame:
    return (
        res_df
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            n=("std_res", "size"),
            mean_res=("std_res", "mean"),
            median_res=("std_res", "median"),
            sd_res=("std_res", "std"),
            mad_res=("std_res", mad),
            q05=("std_res", lambda x: float(np.nanquantile(x, 0.05))),
            q95=("std_res", lambda x: float(np.nanquantile(x, 0.95))),
            frac_abs_gt2=("std_res", lambda x: float(np.mean(np.abs(x) > 2))),
            frac_abs_gt3=("std_res", lambda x: float(np.mean(np.abs(x) > 3))),
            residual_outlier_frac=(
                "is_residual_outlier",
                lambda x: float(np.mean(x)) if len(x) else np.nan,
            )
            if "is_residual_outlier" in res_df.columns
            else (
                "std_res",
                lambda x: float(np.mean(np.abs(x) > 3)),
            ),
        )
        .reset_index()
    )


def residual_x_trend_summary(res_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_step = (
        res_df
        .groupby(["trace_id", "label", "step"], observed=True)
        .agg(
            x_mean=("x", "mean"),
            step_median_res=("std_res", "median"),
            step_mean_res=("std_res", "mean"),
        )
        .reset_index()
    )
    trend = (
        by_step
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            x_median_rms=(
                "step_median_res",
                lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2))),
            ),
            x_median_maxabs=(
                "step_median_res",
                lambda x: float(np.nanmax(np.abs(np.asarray(x)))),
            ),
            x_mean_rms=(
                "step_mean_res",
                lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2))),
            ),
        )
        .reset_index()
    )
    return trend, by_step


def residual_x_correlation(res_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, _t.Any]] = []
    for (trace_id, label), g in res_df.groupby(["trace_id", "label"], observed=True):
        if len(g) < 4:
            continue
        rows.append({
            "trace_id": trace_id,
            "label": label,
            "pearson_res_x": g["std_res"].corr(g["x"], method="pearson"),
            "spearman_res_x": g["std_res"].corr(g["x"], method="spearman"),
        })
    return pd.DataFrame(rows)


def residual_lag1_autocorrelation(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, _t.Any]] = []
    for (trace_id, well, label), g in res_df.groupby(
        ["trace_id", "well", "label"], observed=True
    ):
        g = g.sort_values("step")
        r = g["std_res"].to_numpy(dtype=float)
        if len(r) < 3 or np.nanstd(r[:-1]) == 0 or np.nanstd(r[1:]) == 0:
            lag1 = np.nan
        else:
            lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        rows.append({
            "trace_id": trace_id,
            "well": well,
            "label": label,
            "lag1_res_autocorr": lag1,
        })
    lag_df = pd.DataFrame(rows)
    if lag_df.empty:
        return lag_df, pd.DataFrame()
    summary = (
        lag_df
        .groupby(["trace_id", "label"], observed=True)
        .agg(
            lag1_mean=("lag1_res_autocorr", "mean"),
            lag1_median=("lag1_res_autocorr", "median"),
            lag1_abs_mean=("lag1_res_autocorr", lambda x: float(np.nanmean(np.abs(x)))),
        )
        .reset_index()
    )
    return lag_df, summary


def residual_cross_label_correlation(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = res_df.pivot_table(
        index=["trace_id", "well", "step"],
        columns="label",
        values="std_res",
        aggfunc="mean",
    )
    rows: list[dict[str, _t.Any]] = []
    for trace_id, sub in wide.groupby(level=0):
        mat = sub.droplevel(0)
        labels = list(mat.columns)
        corr = mat.corr()
        for a, b in itertools.combinations(labels, 2):
            rows.append({
                "trace_id": trace_id,
                "label_a": str(a),
                "label_b": str(b),
                "cross_label_corr": float(np.asarray(corr.loc[a, b]).ravel()[0]),
            })
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df, pd.DataFrame()
    summary = (
        corr_df
        .groupby("trace_id", observed=True)
        .agg(
            cross_label_corr_abs_mean=(
                "cross_label_corr",
                lambda x: float(np.nanmean(np.abs(x))),
            ),
            cross_label_corr_abs_max=(
                "cross_label_corr",
                lambda x: float(np.nanmax(np.abs(x))),
            ),
        )
        .reset_index()
    )
    return corr_df, summary


def model_residual_score_table(
    res_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model-level and detailed residual summary tables."""
    dist = residual_distribution_summary(res_df)
    trend, by_step = residual_x_trend_summary(res_df)
    corr_x = residual_x_correlation(res_df)
    lag_df, lag_summary = residual_lag1_autocorrelation(res_df)
    cross_df, cross_summary = residual_cross_label_correlation(res_df)

    per_label = dist.merge(trend, on=["trace_id", "label"], how="left")
    if not corr_x.empty:
        per_label = per_label.merge(corr_x, on=["trace_id", "label"], how="left")
    if not lag_summary.empty:
        per_label = per_label.merge(lag_summary, on=["trace_id", "label"], how="left")

    agg_spec: dict[str, tuple[str, _t.Any]] = {
        "residual_mean_abs": ("mean_res", lambda x: float(np.nanmean(np.abs(x)))),
        "residual_median_abs": ("median_res", lambda x: float(np.nanmean(np.abs(x)))),
        "residual_sd_mean": ("sd_res", "mean"),
        "residual_frac_abs_gt2": ("frac_abs_gt2", "mean"),
        "residual_frac_abs_gt3": ("frac_abs_gt3", "mean"),
        "residual_outlier_frac": ("residual_outlier_frac", "mean"),
        "residual_x_median_rms": ("x_median_rms", "mean"),
        "residual_x_median_maxabs": ("x_median_maxabs", "max"),
    }
    if "spearman_res_x" in per_label:
        agg_spec["residual_abs_spearman_x"] = (
            "spearman_res_x",
            lambda x: float(np.nanmean(np.abs(x))) if len(x) else np.nan,
        )
    if "lag1_abs_mean" in per_label:
        agg_spec["residual_lag1_abs_mean"] = ("lag1_abs_mean", "mean")

    model = per_label.groupby("trace_id", observed=True).agg(**agg_spec).reset_index()
    if not cross_summary.empty:
        model = model.merge(cross_summary, on="trace_id", how="left")
    return model, per_label, by_step, lag_df, cross_df
