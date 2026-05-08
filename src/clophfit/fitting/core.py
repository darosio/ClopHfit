"""
Clophfit: Fitting of Cl- binding and pH titration curves.

This module provides a comprehensive suite of tools for analyzing titration data,
particularly for chloride binding and pH titration experiments common in biochemistry,
such as those involving fluorescent probes.

Core Functionality:
-------------------
1.  **Data Modeling**: Implements a 1-site binding model suitable for both
    ligand concentration and pH titrations.

2.  **Spectral Data Processing**:
    -   Processes raw spectral data (e.g., from fluorescence spectroscopy).
    -   Offers two methods for data reduction:

        -   Singular Value Decomposition (SVD) to extract the most significant
            spectral component.
        -   Band integration over a specified wavelength range.

3.  **Curve Fitting**: Provides three distinct fitting backends to determine the
    dissociation constant (K) and other parameters:

    -   **Least-Squares (LM)**: Utilizes the `lmfit` library for robust non-linear
        least-squares minimization. Supports iterative reweighting and outlier
        removal.
    -   **Orthogonal Distance Regression (ODR)**: Employs `odrpack` to account
        for uncertainties in both x and y variables, which is crucial when x-values
        (e.g., pH measurements) have errors.
    -   **Bayesian Modeling (PyMC)**: Implements a hierarchical Bayesian model
        using `pymc`. This approach is powerful for:

        -   Quantifying parameter uncertainties as full posterior distributions.
        -   Modeling errors in x-values as latent variables.
        -   Sharing information between multiple experiments (hierarchical fitting)
            to obtain more robust parameter estimates.

4.  **Result Visualization**: Includes extensive plotting functions to visualize:
    -   Raw and processed spectra.
    -   Fitted curves with confidence intervals.
    -   Diagnostic plots for SVD and Bayesian analyses (e.g., corner plots).
"""


# --- Data Structures for Fit Results ---
# TODO: from gemini

from __future__ import annotations

import copy
import logging
import os
import typing
from sys import float_info

import lmfit  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer  # type: ignore[import-untyped]
from matplotlib import figure

from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import InsufficientDataError
from clophfit.fitting.models import binding_1site
from clophfit.fitting.plotting import (
    COLOR_MAP,
    PlotParameters,
    _create_spectra_canvas,
    plot_autovalues,
    plot_autovectors,
    plot_emcee_k_on_ax,
    plot_fit,
    plot_pca,
    plot_spectra,
    plot_spectra_distributed,
)
from clophfit.fitting.utils import (
    identify_outliers_zscore,
    parse_remove_outliers,
    reweight_from_residuals,
)

from .data_structures import FitResult, SpectraGlobResults

if typing.TYPE_CHECKING:
    from clophfit.clophfit_types import ArrayF


# --- Globals ---
N_BOOT = 20  # To compute fill_between uncertainty.

EMCEE_STEPS = 1800

logger = logging.getLogger(__name__)


# ---- Helpers ----


def _binding_1site_residuals(params: Parameters, ds: Dataset) -> ArrayF:
    """Compute concatenated residuals for multiple datasets for lmfit.

    Optimized version with reduced dictionary lookups and vectorized operations.
    """
    is_ph = ds.is_ph
    K = params["K"].value  # Extract once to avoid repeated access  # noqa: N806

    # Pre-allocate lists for better performance
    residuals_list: list[ArrayF] = []

    for lbl, da in ds.items():
        # Get parameter values once per label
        S0 = params[f"S0_{lbl}"].value  # noqa: N806
        S1 = params[f"S1_{lbl}"].value  # noqa: N806

        # Compute model and residuals in one go
        model = binding_1site(da.x, K, S0, S1, is_ph=is_ph)
        weight = 1 / da.y_err if da.y_err.size > 0 else np.ones_like(da.y)
        residuals_list.append(weight * (da.y - model))

    return np.concatenate(residuals_list)


def _build_params_1site(ds: Dataset) -> Parameters:
    """Initialize lmfit Parameters for a 1-site model from a Dataset.

    Optimized version with reduced list operations and vectorized computations.
    """
    params = Parameters()
    if ds.is_ph:
        params.add("K", min=3, max=11)
    else:
        # epsilon avoids x/K raise x/0 error
        params.add("K", min=float_info.epsilon)

    # Pre-allocate numpy array for better performance
    k_initial_guesses = np.empty(len(ds))

    for i, (lbl, da) in enumerate(ds.items()):
        params.add(f"S0_{lbl}", value=da.y[0])
        params.add(f"S1_{lbl}", value=da.y[-1])
        # Vectorized computation of target and halfway point
        target_y = (da.y[0] + da.y[-1]) * 0.5
        halfway_idx = np.argmin(np.abs(da.y - target_y))
        k_initial_guesses[i] = da.x[halfway_idx]

    params["K"].value = np.mean(k_initial_guesses)
    return params


# ---- Weight estimation ----


def weight_da(da: DataArray, *, is_ph: bool) -> bool:
    """Estimate initial weights for a DataArray by fitting it individually.

    The standard error of the residuals from this initial fit is used as
    the uncertainty (``y_err``) for subsequent weighted fits.

    Parameters
    ----------
    da : DataArray
        The data array to be weighted.
    is_ph : bool
        Whether the titration is pH-based.

    Returns
    -------
    bool
        True if the weighting fit was successful, False otherwise.
    """
    ds = Dataset.from_da(da, is_ph=is_ph)
    params = _build_params_1site(ds)
    if len(params) > len(da.y):
        # Not enough data points to fit; assign default error
        da.y_err = np.ones_like(da.xc)
        return False
    mr = lmfit.minimize(_binding_1site_residuals, params, args=(ds,))
    # Calculate residuals SEM
    sem = np.std(mr.residual, ddof=1) / np.sqrt(len(mr.residual))
    da.y_err = sem if np.isfinite(sem) and sem > 0 else np.array([1.0])
    return True


def weight_multi_ds_titration(ds: Dataset) -> None:
    """Assign weights to all DataArrays within a Dataset.

    Iterates through each `DataArray` in the `Dataset`, calling `weight_da`
    to estimate `y_err`. For any `DataArray` where weighting fails (e.g., due
    to insufficient data), a fallback error is assigned based on the errors
    from successfully fitted arrays.

    Optimized version with reduced set operations and memory allocations.
    """
    failed_labels = []
    successful_max_yerr = -np.inf

    # Single pass: weight and track failures/successes
    for lbl, da in ds.items():
        if not weight_da(da, is_ph=ds.is_ph):
            failed_labels.append(lbl)
        else:
            # Track maximum error from successful fits
            max_err = np.max(da.y_err).item()
            successful_max_yerr = max(successful_max_yerr, max_err)

    # Assign fallback error to failed labels
    if failed_labels and successful_max_yerr > -np.inf:
        fallback_error = successful_max_yerr * 10
        for lbl in failed_labels:
            ds[lbl].y_err = np.full_like(ds[lbl].xc, fallback_error)


def calibrate_noise_from_residuals(
    residuals_signals: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, tuple[float, float, float]]:
    """Estimate (sigma_read, gain, alpha) per label from pooled first-pass residuals.

    Fits the noise model ``sigma^2 = sigma_read^2 + gain*signal + (alpha*signal)^2``
    by minimising
    the squared difference between empirical residual^2 and the model prediction,
    pooling all wells and pH-steps for each label.

    This is a two-pass calibration intended to replace the heuristic
    gain=1/alpha=0 defaults: run a first-pass fit (e.g. huber), collect all
    (residuals, signal) pairs per label, then call this function.  The returned
    parameters can be fed into ``_create_data_array`` via ``noise_gain`` /
    ``noise_alpha`` so that subsequent fits use physically motivated weights.

    Parameters
    ----------
    residuals_signals : dict[str, list[tuple[np.ndarray, np.ndarray]]]
        Keyed by label name.  Each entry is a list of ``(residuals, signal)``
        arrays — one pair per well — where both arrays have the same length
        (one element per pH step that was not masked).

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Per label: ``(sigma_read, gain, alpha)`` — all non-negative.

    Notes
    -----
    When there are too few data points or the optimisation fails, returns
    ``(1.0, 0.0, 0.0)`` as a safe fallback (unweighted).

    Examples
    --------
    >>> import numpy as np
    >>> residuals_signals = {
    ...     "y1": [(np.array([3.0, 4.0, 5.0]), np.array([100.0, 200.0, 300.0]))]
    ... }
    >>> params = calibrate_noise_from_residuals(residuals_signals)
    >>> "y1" in params
    True
    """
    result: dict[str, tuple[float, float, float]] = {}

    for lbl, pairs in residuals_signals.items():
        if not pairs:
            result[lbl] = (1.0, 0.0, 0.0)
            continue

        res_all = np.concatenate([r for r, _ in pairs])
        sig_all = np.concatenate([s for _, s in pairs])

        if len(res_all) < 4:  # noqa: PLR2004
            result[lbl] = (1.0, 0.0, 0.0)
            continue

        sq_res = res_all**2
        sig_all**2

        # Noise model: var = a + b*signal + c*signal^2
        # a = sigma_read^2, b = gain, c = alpha^2
        def noise_model(signal: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a + b * signal + c * signal**2

        p0 = [np.median(sq_res), 0.0, 0.0]
        try:
            import scipy.optimize as sco  # noqa: PLC0415

            popt, _ = sco.curve_fit(
                noise_model,
                sig_all,
                sq_res,
                p0=p0,
                bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
                maxfev=10000,
            )
            a_opt, b_opt, c_opt = popt
            result[lbl] = (float(np.sqrt(a_opt)), float(b_opt), float(np.sqrt(c_opt)))
        except Exception:  # noqa: BLE001
            # fallback: use residual RMS as flat sigma_read
            result[lbl] = (float(np.sqrt(np.mean(sq_res))), 0.0, 0.0)

    return result


def weight_multi_ds_calibrated(
    datasets: dict[str, Dataset],
    *,
    is_ph: bool,
) -> dict[str, tuple[float, float, float]]:
    """Two-pass noise calibration: fit all wells, then assign signal-dependent weights.

    Runs a first-pass least-squares fit on every well, pools (residual, signal)
    pairs across all wells for each label, calibrates the noise model
    ``sigma^2 = sigma_read^2 + gain*signal + (alpha*signal)^2`` via
    :func:`calibrate_noise_from_residuals`, and then re-assigns
    ``DataArray.y_err`` using the calibrated, signal-dependent formula.

    Parameters
    ----------
    datasets : dict[str, Dataset]
        Keyed by well identifier; each Dataset contains one or more labels.
        *Modified in-place*: ``da.y_err`` is overwritten for every DataArray.
    is_ph : bool
        Passed to the first-pass fit.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Per label: ``(sigma_read, gain, alpha)`` as estimated from the pooled
        residuals.  Useful for inspecting / logging the calibrated parameters.

    Notes
    -----
    Wells that fail the first-pass fit are excluded from calibration but still
    get weights via fallback (10x the maximum calibrated ``sigma_read``).
    """
    # Step 1: first-pass fits — collect (residuals, signal) per label
    residuals_signals: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for well, ds in datasets.items():
        for lbl, da in ds.items():
            well_ds = Dataset.from_da(da, is_ph=is_ph)
            params = _build_params_1site(well_ds)
            if len(params) >= len(da.y):
                continue
            try:
                mr = lmfit.minimize(_binding_1site_residuals, params, args=(well_ds,))
                resid = np.asarray(mr.residual, dtype=float)
                sig = np.asarray(da.y, dtype=float)
                residuals_signals.setdefault(lbl, []).append((resid, sig))
            except Exception:  # noqa: BLE001
                logger.debug("First-pass fit failed for well %s label %s", well, lbl)

    # Step 2: calibrate noise model across all wells
    noise_params = calibrate_noise_from_residuals(residuals_signals)

    # Step 3: assign signal-dependent y_err to each DataArray
    for ds in datasets.values():
        for lbl, da in ds.items():
            sr, gain, alpha = noise_params.get(lbl, (1.0, 0.0, 0.0))
            signal = np.asarray(da.y, dtype=float)
            sigma = np.sqrt(sr**2 + gain * np.abs(signal) + (alpha * signal) ** 2)
            da.y_err = np.where(sigma > 0, sigma, 1.0)

    return noise_params


def analyze_spectra(
    spectra: pd.DataFrame, *, is_ph: bool, band: tuple[int, int] | None = None
) -> FitResult[Minimizer]:
    """Analyze spectra titration, fit the data, and plot the results.

    This function performs either Singular Value Decomposition (SVD) or
    integrates spectra over a specified band.

    Parameters
    ----------
    spectra : pd.DataFrame
        The DataFrame containing spectra (one spectrum for each column).
    is_ph : bool
        Whether the x-axis represents pH.
    band : tuple[int, int] | None
        If provided, use the 'band' integration method. Otherwise, use 'svd'.

    Returns
    -------
    FitResult[Minimizer]
        An object containing the fit results and the summary plot.

    Raises
    ------
    ValueError
        If the band parameters are not in the spectra's index when the band
        method is used.

    Notes
    -----
    Creates plots of spectra, principal component vectors, singular values, fit
    of the first principal component and PCA for SVD; only of spectra and fit
    for Band method.
    """
    y_offset = 1.0
    x = spectra.columns.to_numpy()
    fig, (ax1, ax2, ax3, ax4, ax5) = _create_spectra_canvas()
    plot_spectra(ax1, spectra, PlotParameters(is_ph))
    if band is None:  # SVD
        threshold = int(len(spectra) * 0.5)
        spectra = spectra.dropna(axis=1, thresh=threshold)
        spectra = spectra.dropna(axis=0)
        ddf = spectra.sub(spectra.iloc[:, 0], axis=0)
        u, s, v = np.linalg.svd(ddf)
        ds = Dataset({"default": DataArray(x, v[0, :] + y_offset)}, is_ph=is_ph)
        plot_autovectors(ax2, spectra.index, u)
        plot_autovalues(ax3, s[:])  # don't plot last auto-values?
        plot_pca(ax5, v, x, PlotParameters(is_ph))
        ylabel = "First Principal Component"
        ylabel_color = COLOR_MAP(0)
    else:  # Band integration
        ini, fin = band
        if ini not in spectra.index and fin not in spectra.index:
            msg = f"Band parameters ({ini}, {fin}) are not in the spectra's index."
            raise ValueError(msg)
        y = np.array([
            spectra.iloc[:, i].loc[ini:fin].sum() for i in range(spectra.shape[1])
        ])
        y /= np.abs(y).max() / 10
        ds = Dataset.from_da(DataArray(x, y), is_ph=is_ph)
        ylabel = "Integrated Band Fluorescence"
        ylabel_color = (0.0, 0.0, 0.0, 1.0)

    weight_multi_ds_titration(ds)
    fit_result = fit_binding_glob(ds)
    result = fit_result.result
    mini = fit_result.mini
    fit_params = result.params if result else Parameters()
    plot_fit(ax4, ds, fit_params, nboot=N_BOOT, pp=PlotParameters(is_ph))
    ax4.set_ylabel(ylabel, color=ylabel_color)
    return FitResult(fig, result, mini, ds)


def analyze_spectra_glob(
    titration: dict[str, pd.DataFrame],
    ds: Dataset,
    dbands: dict[str, tuple[int, int]] | None = None,
) -> SpectraGlobResults:
    """Analyze multi-label spectra visualize the results."""
    _gap_ = 1
    dbands = dbands or {}
    labels_svd = titration.keys() - dbands.keys()
    labels_bands = titration.keys() - labels_svd
    if len(labels_svd) > 1:
        prev_max = 0
        adjusted_list: list[pd.DataFrame] = []
        for lbl in labels_svd:
            spectra_adjusted = titration[lbl].copy()
            spectra_adjusted.index += prev_max - spectra_adjusted.index.min() + _gap_
            prev_max = spectra_adjusted.index.max()
            adjusted_list.append(spectra_adjusted)
        spectra_merged = pd.concat(adjusted_list)
        svd = analyze_spectra(spectra_merged, is_ph=ds.is_ph)
        ds_svd = ds.copy(labels_svd)
        weight_multi_ds_titration(ds_svd)
        f_res = fit_binding_glob(ds_svd)
        fig = _plot_spectra_glob_emcee(titration, ds_svd, f_res)
        gsvd = FitResult(fig, f_res.result, f_res.mini)
    else:
        svd, gsvd = None, None

    if len(labels_bands) > 1:
        ds_bands = ds.copy(labels_bands)
        weight_multi_ds_titration(ds_bands)
        f_res = fit_binding_glob(ds_bands)
        fig = _plot_spectra_glob_emcee(titration, ds_bands, f_res, dbands)
        bands = FitResult(fig, f_res.result, f_res.mini)
    else:
        bands = None

    return SpectraGlobResults(svd, gsvd, bands)


def _plot_spectra_glob_emcee(
    titration: dict[str, pd.DataFrame],
    ds: Dataset,
    f_res: FitResult[Minimizer],
    dbands: dict[str, tuple[int, int]] | None = None,
) -> figure.Figure:
    fig, (ax1, ax2, ax3, ax4, ax5) = _create_spectra_canvas()
    fig.delaxes(ax1)
    fig.delaxes(ax2)
    fig.delaxes(ax3)
    pparams = PlotParameters(ds.is_ph)
    tit_filtered = {k: spec for k, spec in titration.items() if k in ds}
    plot_spectra_distributed(fig, tit_filtered, pparams, dbands)
    fit_params = f_res.result.params if f_res.result else Parameters()
    plot_fit(ax4, ds, fit_params, nboot=N_BOOT, pp=pparams)
    if f_res.mini is not None:
        workers = int(os.environ.get("CLOPHFIT_EMCEE_WORKERS", "4"))
        result_emcee = f_res.mini.emcee(
            steps=EMCEE_STEPS * 3,
            workers=workers,
            burn=100,
            nwalkers=30,
            progress=False,
        )
        plot_emcee_k_on_ax(ax5, result_emcee)
    return fig


# ---- Unified LM fitting function ----


def fit_binding_glob(  # noqa: PLR0913
    ds: Dataset,
    *,
    method: str = "lm",
    reweight: str | None = None,
    remove_outliers: str | None = None,
    max_iter: int = 15,
    tol: float = 0.01,
    scale_covar: bool = True,
) -> FitResult[Minimizer]:
    r"""Analyze multi-label titration datasets and visualize the results.

    Unified fitting function that supports standard least-squares and robust
    fitting with optional iterative reweighting and outlier detection.

    Parameters
    ----------
    ds : Dataset
        Input dataset with *x*, *y*, and *y_err* for each label.
    method : str, optional
        Fitting method: ``"lm"`` (default) for standard least-squares or
        ``"huber"`` for Huber-loss robust fitting (reduces outlier influence).
    reweight : str | None, optional
        Reweighting strategy to apply **after** each residual evaluation:

        * ``"irls"`` - iteratively reweighted least-squares (uniform
          scale per label from MA-residual).

        Default is ``None`` (no reweighting).
    remove_outliers : str | None, optional
        Outlier-removal specification of the form ``"zscore:threshold:min_keep"``
        where *threshold* is the z-score cutoff and *min_keep* is the minimum
        number of points required per label.  Default is ``None``.
    max_iter : int, optional
        Maximum number of iterations for iterative procedures (reweighting).
        Default is 15.
    tol : float, optional
        Convergence tolerance on the reduced chi-squared.  The loop stops when
        the improvement drops below this value.  Default is 0.01.
    scale_covar : bool, optional
        Whether to scale the covariance matrix.  Default is ``True``.

    Returns
    -------
    FitResult[Minimizer]
        An object containing the fit results, plot figure, minimizer, and
        dataset copy.

    Raises
    ------
    InsufficientDataError
        If there are not enough data points for the number of parameters.

    Notes
    -----
    Parameter uncertainties are scaled by :math:`\\sqrt{\\chi^2_\\nu}` via
    lmfit's Minimizer(scale_covar=True), which improves coverage when errors
    are underestimated.

    Residuals returned are WEIGHTED (weight * (observed - predicted)) where
    weight = 1/y_err.  This is appropriate for heteroscedastic data where
    different observations have different uncertainties.
    """
    params = _build_params_1site(ds)
    total_len = sum(len(da.y) for da in ds.values())
    if len(params) > total_len:
        msg = "Not enough data points for the number of parameters."
        raise InsufficientDataError(msg)

    mini = Minimizer(
        _binding_1site_residuals,
        params,
        fcn_args=(ds,),
        scale_covar=scale_covar,
    )
    minimize_kwargs: dict[str, str] = {"method": "least_squares"}
    if method == "huber":
        minimize_kwargs["loss"] = "huber"

    # Iterative reweighting loop
    ds_working: Dataset = ds
    result = mini.minimize(**minimize_kwargs)
    if reweight is not None and result is not None:
        resid_var = result.redchi
        for _i in range(max_iter):
            ds_working = reweight_from_residuals(ds_working, result.residual)

            mini = Minimizer(
                _binding_1site_residuals,
                _build_params_1site(ds_working),
                fcn_args=(ds_working,),
                scale_covar=scale_covar,
            )
            new_res = mini.minimize(**minimize_kwargs)
            if new_res is None or new_res.redchi == 0:
                result = new_res
                break
            if resid_var - new_res.redchi < tol:
                result = new_res
                break
            resid_var = new_res.redchi
            result = new_res

    # Outlier masking
    if remove_outliers is not None and result is not None:
        method_name, z_threshold, _min_keep = parse_remove_outliers(remove_outliers)
        current_len = sum(len(da.y) for da in ds_working.values())
        combined_mask = np.ones(current_len, dtype=bool)
        start_idx = 0
        for da in ds_working.values():
            end_idx = start_idx + len(da.y)
            dr = result.residual[start_idx:end_idx]
            if method_name == "zscore":
                outliers = identify_outliers_zscore(dr, threshold=z_threshold)
                combined_mask[start_idx:end_idx] &= ~outliers
            start_idx = end_idx
        ds_filtered = copy.deepcopy(ds_working)
        ds_filtered.apply_mask(combined_mask)
        ds_working = ds_filtered
        params = _build_params_1site(ds_working)
        mini = Minimizer(
            _binding_1site_residuals,
            params,
            fcn_args=(ds_working,),
            scale_covar=scale_covar,
        )
        result = mini.minimize(**minimize_kwargs)

    fig = figure.Figure()
    ax = fig.add_subplot(111)
    fit_params = result.params if result else Parameters()
    plot_fit(
        ax, ds_working, fit_params, nboot=N_BOOT, pp=PlotParameters(ds_working.is_ph)
    )
    return FitResult(fig, result, mini, copy.deepcopy(ds_working))


# ---- Legacy helper ----


# ---- Public API ----

__all__ = [
    # Helpers
    "_binding_1site_residuals",
    "_build_params_1site",
    # Spectral analysis
    "analyze_spectra",
    "analyze_spectra_glob",
    # Fitting
    "fit_binding_glob",
    # Weight estimation
    "weight_da",
    "weight_multi_ds_titration",
]
