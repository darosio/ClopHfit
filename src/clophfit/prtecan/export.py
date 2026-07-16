"""Export data and fit results from Titration objects."""

import copy
import itertools
import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from clophfit.clophfit_types import ArrayF
from clophfit.fitting.bayes import fit_binding_pymc, fit_binding_pymc_residual_refit
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.data_structures import FitResult
from clophfit.fitting.diagnostics import detect_bad_wells
from clophfit.fitting.model_validation import residuals_from_fit_results
from clophfit.fitting.models import binding_1site
from clophfit.fitting.pipeline import fit_plate
from clophfit.fitting.residuals import (
    plot_residual_vs_predicted,
    plot_residual_vs_yerr,
    residual_statistics,
)
from clophfit.prtecan.titration import TecanConfig, Titration, TitrationResults

logger = logging.getLogger(__name__)


def generate_combinations() -> list[tuple[tuple[bool, ...], str]]:
    """Generate parameter combinations for export and fitting."""
    bool_iter = itertools.product([False, True], repeat=4)
    return [
        (tuple(bool_combo), method)
        for bool_combo in bool_iter
        for method in ["mean", "meansd", "fit"]
    ]


def apply_combination(
    titration: Titration, combination: tuple[tuple[bool, ...], str]
) -> None:
    """Apply a combination of parameters to the Titration."""
    (bg, adj, dil, nrm), method = combination
    logger.info("Params are: ........... %s", ((bg, adj, dil, nrm), method))
    titration.params.bg = bg
    titration.params.bg_adj = adj
    titration.params.dil = dil
    titration.params.nrm = nrm
    titration.params.bg_mth = method


def prepare_output_folder(titration: Titration, base_path: Path) -> Path:
    """Prepare the output folder for a given combination of parameters."""
    p = titration.params
    sbg = "_bg" if p.bg else ""
    sadj = "_adj" if p.bg_adj else ""
    sdil = "_dil" if p.dil else ""
    snrm = "_nrm" if p.nrm else ""
    sfit = "_fit" if p.bg_mth == "fit" else ""
    smeansd = "_1sd" if p.bg_mth == "meansd" else ""
    subfolder_name = "dat" + sbg + sadj + sdil + snrm + sfit + smeansd
    subfolder_path = base_path / subfolder_name
    subfolder_path.mkdir(parents=True, exist_ok=True)
    return subfolder_path


def export_residuals(
    outfit: Path, fit_results: dict[str, FitResult[typing.Any]], index: int
) -> None:
    """Export fit residuals and their statistics to files."""
    try:
        all_res = residuals_from_fit_results(
            fit_results, trace_id="", binding_function=binding_1site
        )
    except (ValueError, KeyError):
        return
    all_res.to_csv(outfit / f"residuals_{index}.csv", index=False)
    stats = residual_statistics(all_res)
    stats.to_csv(outfit / f"residual_stats_{index}.csv")
    label = str(index)
    fig_pred = plot_residual_vs_predicted(all_res, title=label)
    fig_pred.savefig(outfit / f"residual_vs_predicted_{index}.png", dpi=150)
    fig_yerr = plot_residual_vs_yerr(all_res, title=label)
    fig_yerr.savefig(outfit / f"residual_vs_yerr_{index}.png", dpi=150)


def export_bad_wells(outfit: Path, global_res: TitrationResults) -> None:
    """Export a list of bad wells flagged during global fits."""
    df_fit = global_res.dataframe.reset_index(names="well")
    df = detect_bad_wells(df_fit)
    bad = df[df["flag_any"]].sort_values("flag_count", ascending=False)
    bad.to_csv(outfit / "bad_wells.csv", index=False)


def run_pre_fit_detection(titration: Titration, subfolder: Path) -> None:
    """Run pre-fit detection of bad wells and write discarded_wells.txt."""
    discards = titration.detect_and_discard_bad_wells()
    if discards:
        logger.info("Pre-fit bad-well detection: discarding %s", discards)
        (subfolder / "discarded_wells.txt").write_text(
            "\n".join(sorted(discards)) + "\n", encoding="utf-8"
        )


def fit_single_mcmc(
    titration: Titration,
    datasets: dict[str, typing.Any],
    outfit: Path,
) -> TitrationResults | None:
    """Run optional per-well single PyMC fits for export.

    Parameters
    ----------
    titration : Titration
        Titration object containing the plate scheme, fit keys, background
        noise, and MCMC configuration.
    datasets : dict[str, typing.Any]
        Mapping from well identifiers to datasets to fit.
    outfit : Path
        Output directory used for residual-refit diagnostic CSV files.

    Returns
    -------
    TitrationResults | None
        Per-well PyMC fit results when ``titration.params.mcmc`` is
        ``"single"`` or ``"single-refit"``. Returns ``None`` when single-well
        MCMC export is disabled.
    """
    if titration.params.mcmc == "single":
        mcmc_fits = {
            key: fit_binding_pymc(
                ds,
                sampler=SamplerConfig(
                    n_samples=titration.params.n_mcmc_samples,
                    nuts_sampler=titration.params.nuts_sampler,
                ),
            )
            for key, ds in datasets.items()
        }
        return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)

    if titration.params.mcmc != "single-refit":
        return None

    mcmc_fits = {}
    residual_rows = []
    for key, ds in datasets.items():
        refit = fit_binding_pymc_residual_refit(
            ds,
            bg_noise=titration.bg_noise,
            n_samples=titration.params.n_mcmc_samples,
            nuts_sampler=titration.params.nuts_sampler,
        )
        mcmc_fits[key] = refit.final
        if not refit.residuals.empty:
            residual_rows.append(refit.residuals.assign(well=key))
    if residual_rows:
        pd.concat(residual_rows, ignore_index=True).to_csv(
            outfit / "single_refit_initial_residual_outliers.csv", index=False
        )
    return TitrationResults(titration.scheme, titration.fit_keys, mcmc_fits)


def export_fit(titration: Titration, subfolder: Path, config: TecanConfig) -> None:
    """Export all fitted parameters, plots, and data files."""
    outfit = subfolder / "fit"
    outfit.mkdir(parents=True, exist_ok=True)

    datasets = {k: titration.create_global_ds(k) for k in titration.fit_keys}

    export_list = []
    for label, dat in titration.data.items():
        if dat:
            ds_single = {
                k: titration.create_ds(k, label=label) for k in titration.fit_keys
            }
            fits = fit_plate(
                ds_single,
                method=titration.params.fit_method,
                remove_outliers=titration.params.outlier,
            )
            export_list.append(
                TitrationResults(titration.scheme, titration.fit_keys, fits)
            )

    method = (
        "lm"
        if titration.params.fit_method == "irls"
        else ("huber" if titration.params.fit_method == "huber" else "lm")
    )
    reweight = "irls" if titration.params.fit_method == "irls" else None

    global_fits = fit_plate(
        datasets,
        method=method,
        reweight=reweight,
        remove_outliers=titration.params.outlier,
    )
    global_res = TitrationResults(titration.scheme, titration.fit_keys, global_fits)
    export_list.append(global_res)

    odr_fits = fit_plate(
        datasets,
        method="odr",
        remove_outliers=titration.params.outlier,
        reweight=reweight,
    )
    odr_res = TitrationResults(titration.scheme, titration.fit_keys, odr_fits)
    export_list.append(odr_res)

    mcmc_res = fit_single_mcmc(titration, datasets, outfit)
    if mcmc_res is not None:
        export_list.append(mcmc_res)

    for i, results in enumerate(export_list):
        png_dir = outfit / f"lb{i}"
        data_dir = png_dir / "ds"
        for key in results.fit_keys:
            fr = results[key]
            if config.png:
                if fr.figure:
                    png_dir.mkdir(parents=True, exist_ok=True)
                    fr.figure.savefig(png_dir / f"{key}.png")
                if fr.dataset:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    fr.dataset.export(data_dir / f"{key}.csv")
        fit = results.dataframe
        fit.sort_index().to_csv(outfit / f"ffit{i}.csv")
        title = config.title + f"lb:{i}"
        f = results.plot_k(xlim=config.lim, title=title)
        f.savefig(outfit / f"K{i}.png")
        export_residuals(outfit, results.results, i)

    if config.detect_bad:
        export_bad_wells(outfit, global_res)


def export_data_fit(titration: Titration, tecan_config: TecanConfig) -> None:
    """Export dat files [x,y1,..,yN] from copy of titration.data."""

    def write(x: ArrayF, data: dict[str, dict[str, ArrayF]], out_folder: Path) -> None:
        if any(data):
            out_folder.mkdir(parents=True, exist_ok=True)
            columns = ["x"] + [str(i) for i in data]
            first_label = next(iter(titration.labelblocksgroups.keys()))
            for key in data[first_label]:
                dat = np.vstack((x, [data[i][key] for i in data]))
                datxy = pd.DataFrame(dat.T, columns=columns)
                datxy.to_csv(out_folder / f"{key}.dat", index=False)

    if tecan_config.comb:
        saved_p = copy.copy(titration.params)
        combinations = generate_combinations()
        for combination in combinations:
            apply_combination(titration, combination)
            subfolder = prepare_output_folder(titration, tecan_config.out_fp)
            write(titration.x, titration.data, subfolder)
            if tecan_config.fit:
                if tecan_config.detect_bad:
                    run_pre_fit_detection(titration, subfolder)
                export_fit(titration, subfolder, tecan_config)
        titration.params = saved_p
    else:
        subfolder = prepare_output_folder(titration, tecan_config.out_fp)
        write(titration.x, titration.data, subfolder)
        if tecan_config.fit:
            if tecan_config.detect_bad:
                run_pre_fit_detection(titration, subfolder)
            export_fit(titration, subfolder, tecan_config)
