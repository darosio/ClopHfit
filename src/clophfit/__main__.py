"""Command-line interface."""

from __future__ import annotations

import csv
import json
import logging
import os
import pprint
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import click
import lmfit  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click import Context, Path as cPath

from clophfit.fitting.grid import model_signature

# Set unique pytensor compile dir per process to avoid atexit filelock race
# when multi-chain MCMC spawns 4 subprocesses sharing ~/.pytensor/.lock.
# Also apply optimizer=fast_compile so nutpie does not hit the loop-fusion
# kernel argument limit on large plate models (e.g. 88 wells x 6 pH steps).
_pt_flags = os.environ.get("PYTENSOR_FLAGS", "")
_pt_additions = []
if "base_compiledir" not in _pt_flags:
    _pt_additions.append(f"base_compiledir=/tmp/pytensor_{os.getpid()}")
if "optimizer" not in _pt_flags:
    _pt_additions.append("optimizer=fast_compile")
if _pt_additions:
    os.environ["PYTENSOR_FLAGS"] = ",".join(filter(None, [_pt_flags, *_pt_additions]))
# Force JAX (blackjax/numpyro) onto CPU to avoid GPU OOM with vectorised chains
# on large plate models (~88 wells).  Users can override by setting
# JAX_PLATFORM_NAME before invoking ppr.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from clophfit import (
    __enspire_out_dir__,
    __tecan_out_dir__,
    configure_logging,
    fitting,
)
from clophfit.fitting.bayes_config import RobustConfig, SamplerConfig
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.errors import (
    DataValidationError,
    MissingDependencyError,
)
from clophfit.fitting.spectral import (
    fit_spectra_global_bayes,
    fit_spectra_global_bayes_marginal,
)
from clophfit.prenspire import (
    EnspireFile,
    Note,
    bands as bands_module,
    merge_enspire_files,
)
from clophfit.prenspire.run import (
    BandSummary,
    EnspireInputError,
    EnspireMessage,
    EnspireRunError,
    EnspireRunResult,
    GlobalSummary,
    result_report,
    run_bands,
    run_bayes,
    run_global,
    run_global_bayes,
    run_svd,
    run_vanthoff,
    run_vanthoff_bayes,
)
from clophfit.prenspire.spectral import (
    isosbestic_by_group,
    plate_isosbestic,
    sample_spectra,
)
from clophfit.prtecan import McmcSpec, TecanConfig, Titration, calculate_conc
from clophfit.prtecan.export import export_data_fit

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FlexChoice(click.Choice):
    """click.Choice that also accepts underscores in place of hyphens."""

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        """Normalize underscores to hyphens before validation."""
        normalized = value.replace("_", "-")
        return cast("str", super().convert(normalized, param, ctx))


# Default for --bg-mth. Choosing a non-default method means wanting the
# background subtracted, so it implies --bg: otherwise `--bg-mth fit` picks a
# background that is then never used, and the only clue is the output folder
# quietly losing its `_bg` suffix.
_DEFAULT_BG_MTH = "mean"


@click.group()
@click.version_option(message="%(version)s")
def clop() -> None:  # pragma: no cover
    """Group command."""


def _echo_spec(spec: dict[str, Any]) -> None:
    """Print a resolved analysis specification and the signature it hashes to.

    Parameters
    ----------
    spec : dict[str, Any]
        Resolved option values, one per factor the analysis varies.

    Notes
    -----
    The option surface is large enough that two invocations can describe the
    same analysis while looking different, and different analyses can look
    alike. The signature is taken over resolved values, so it answers "have I
    run this before" without relying on anyone remembering which flags they
    typed.
    """
    signature = model_signature(spec, (), tuple(spec))
    width = max(len(k) for k in spec)
    for key, value in spec.items():
        click.echo(f"  {key:<{width}}  {value!r}")
    click.echo(f"\nsignature: {signature}")


@clop.command()
@click.argument("kd1", type=float)
@click.argument("pka", type=float)
@click.argument("ph", type=float)
def eq1(kd1: float, pka: float, ph: float) -> None:
    """Model Kd dependence on pH."""
    click.echo(fitting.models.kd(kd1=kd1, pka=pka, ph=ph))


@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Increase verbosity: -v for INFO, -vv for DEBUG. Default is WARNING.")  # fmt: skip
@click.option("--quiet", "-q", is_flag=True, help="Silence terminal output; show only ERROR messages.")  # fmt: skip
@click.option("--out", "-o", type=cPath(), help="Output folder.")
def ppr(ctx: Context, verbose: int, quiet: bool, out: str) -> None:  # pragma: no cover
    """Parse Plate Reader `ppr` group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["QUIET"] = quiet
    if out:
        ctx.obj["OUT"] = out


######################################
# pr.tecan                           #
######################################
@ppr.command()
@click.pass_context
@click.argument("list_file", type=cPath(exists=True))
@click.option("--cl", type=float, help="Cl stock concentration (mM) of added aliquots.")
@click.option("--bg", is_flag=True, help="Subtract buffer signal (from scheme.txt).  Implied by --bg-adj.")  # fmt: skip
@click.option("--bg-adj", is_flag=True, help="Heuristically adjust negative buffer values (implies --bg).")  # fmt: skip
@click.option("--bg-mth", default=_DEFAULT_BG_MTH, show_default=True, type=click.Choice(["mean", "median", "fit", "meansd", "mediansd"]), help="Buffer calculation method.")  # fmt: skip
@click.option("--nrm", is_flag=True, help="Normalize using label metadata.")
@click.option("--raw-dir", type=cPath(exists=True, file_okay=False), help="Folder holding the Tecan .xls files, when they are not next to LIST_FILE.")  # fmt: skip
@click.option("--label", multiple=True, help="Fit only these measurement labels (repeatable), e.g. --label 2 for the 485 nm excitation channel alone. A plate carries 400 nm as label 1 and 485 nm as label 2; the 400 nm channel turns over at the acidic end, which one pKa cannot describe, and on this campaign's controls including it costs accuracy against the bench pK (median |error| 0.089 against 0.055 pH) and replicate agreement (0.225 against 0.195). Unset fits every label, as before.")  # fmt: skip
@click.option("--sch", type=cPath(exists=True), help="Path to plate scheme file (buffers and controls).")  # fmt: skip
@click.option("--add", type=cPath(exists=True), help="Path to additions file (initial volume + additions); enables dilution correction.")  # fmt: skip
@click.option("--all", "comb", is_flag=True, help="Export all bg/dil/nrm data combinations.")  # fmt: skip
@click.option("--lim", type=(float, float), help="x-axis range MIN MAX for K plots.")
@click.option("--title", "-t", type=str, default="", help="Title for plots.")
@click.option("--fit/--no-fit", default=True, show_default=True, help="Perform fitting.")  # fmt: skip
@click.option("--png/--no-png", default=True, show_default=True, help="Export PNG files.")  # fmt: skip
@click.option("--fit-method", default="huber", show_default=True, type=click.Choice(["lm", "huber", "irls", "odr"], case_sensitive=False), help="Global fit method: lm (standard LS), huber (robust Huber loss), irls (iterative reweighting), odr (orthogonal distance regression, x-aware).")  # fmt: skip
@click.option("--fit-noise", type=click.Choice(["fixed", "gain", "floor-gain"], case_sensitive=False), default="fixed", show_default=True, help="Weights for the per-well global fit (--fit-method lm or huber). fixed uses y_err as built. gain weights by floor^2 + gain*yhat with alpha 0: every well is fitted, the dof-corrected residuals of all wells are pooled into one gain per label, and the wells are refitted until it settles; floor-gain calibrates the floor too. Writes noise_single_history.csv.")  # fmt: skip
@click.option("--outlier", default=None, type=str, help="Outlier removal spec, e.g. 'mad:3.5:4' (method:threshold:min_keep).")  # fmt: skip
@click.option("--mcmc", type=_FlexChoice(["None", "single", "single-refit", "multi"], case_sensitive=False), default="None", show_default=True, help="MCMC sampling: None, single, single-refit (robust screening pass then refit), multi (all wells jointly, control K shared per group).")  # fmt: skip
@click.option("--nuts-sampler", type=click.Choice(["default", "blackjax", "numpyro", "nutpie"], case_sensitive=False), default="default", show_default=True, help="NUTS backend: default (pytensor/CPU), blackjax/numpyro (JAX/CPU), nutpie (Rust/CPU).")  # fmt: skip
@click.option("--mcmc-samples", default=2000, show_default=True, type=int, help="Number of posterior draws per chain (tune = samples // 2).")  # fmt: skip
@click.option("--noise-alpha", multiple=True, type=float, default=(), help="Proportional noise coefficient per label. Adds proportional term to y_err variance. Obtain from MCMC multi-noise shared_noise_params.csv.")  # fmt: skip
@click.option("--noise-gain", multiple=True, type=float, default=(), help="Poisson gain per label. Replaces hardcoded gain=1 in shot-noise term. Obtain from MCMC multi-noise shared_noise_params.csv.")  # fmt: skip
@click.option("--noise-floor", multiple=True, type=float, default=(), help="Read-noise floor per label, overriding the measured bg_read_noise. Reaches y_err, the FGLS/NNLS calibration and the --mcmc-noise structured floor alike. Use it to apply a floor pooled across plates rather than one estimated from a single plate's three to six buffer wells.")  # fmt: skip
@click.option("--noise-floor-ref-gain", multiple=True, type=float, default=(), help="Reader Gain each --noise-floor was quoted at, per label. The floor is then scaled to this plate's own Gain by 10**((gain-ref)/34.1), so one calibration serves plates read at different settings. Pass 0 to leave a label's floor unscaled, which is right where no Gain dependence was measured.")  # fmt: skip
@click.option("--mcmc-noise", type=click.Choice(["ye_mag", "structured"], case_sensitive=False), default="ye_mag", show_default=True, help="Observation-noise family for --mcmc single-refit and multi. ye_mag scales y_err by a learned multiplier; structured builds floor+gain*y+(alpha*y)^2 with floors from bg_noise and gain/alpha from --noise-gain/--noise-alpha.")  # fmt: skip
@click.option("--noise-mode", type=click.Choice(["centered", "fixed"], case_sensitive=False), default="centered", show_default=True, help="For --mcmc-noise structured, how a hint is treated: centered (a prior the posterior may leave) or fixed (pinned). It governs the floor, which is always hinted because it comes from the measured buffer noise, and any --noise-gain/--noise-alpha supplied. Gain and alpha given no value stay free, so --noise-mode fixed on its own means a pinned floor with both signal-dependent terms learned.")  # fmt: skip
@click.option("--noise-floor-mode", type=click.Choice(["centered", "fixed"], case_sensitive=False), default=None, help="Override --noise-mode for floor alone. One mode for all three cannot separate the terms: pinning alpha at zero also pins the floor, so sigma cannot rescale and the run measures that instead of the term it meant to isolate.")  # fmt: skip
@click.option("--noise-gain-mode", type=click.Choice(["centered", "fixed"], case_sensitive=False), default=None, help="Override --noise-mode for gain alone. One mode for all three cannot separate the terms: pinning alpha at zero also pins the floor, so sigma cannot rescale and the run measures that instead of the term it meant to isolate.")  # fmt: skip
@click.option("--noise-alpha-mode", type=click.Choice(["centered", "fixed"], case_sensitive=False), default=None, help="Override --noise-mode for alpha alone. One mode for all three cannot separate the terms: pinning alpha at zero also pins the floor, so sigma cannot rescale and the run measures that instead of the term it meant to isolate.")  # fmt: skip
@click.option("--noise-ye-mag/--no-noise-ye-mag", "noise_ye_mag", default=False, show_default=True, help="For --mcmc-noise structured: also learn a ye_mag multiplier on sigma, per label (or per well with --per-well-ye-mags). A structured model otherwise has no overall multiplier, so a supplied floor/gain sets the level as well as the shape; this separates the two.")  # fmt: skip
@click.option("--per-well-ye-mags/--no-per-well-ye-mags", "per_well_ye_mags", default=None, help="For --mcmc multi: scale y_err per well rather than per label. Unset lets the library resolve it from the noise family, which couples the two.")  # fmt: skip
@click.option("--ye-mag-parameterization", type=click.Choice(["centered", "hierarchical", "separable", "separable_step"], case_sensitive=False), default="centered", show_default=True, help="For --mcmc multi with per-well ye_mags: independent per label (centered), a shared well factor with per-label deviations (hierarchical), a per-label level plus one shared well factor (separable), or that plus a per-label pH axis on the noise (separable_step).")  # fmt: skip
@click.option("--plate-fit", type=click.Choice(["lm", "odr"], case_sensitive=False), default=None, help="Also fit the whole plate in one classical least-squares problem, with the noise scale profiled per label across the plate and each control group pooled onto one K. Writes plate_{method}_K.csv. Minutes rather than hours, and as accurate against known pKs as the sampler.")  # fmt: skip
@click.option("--plate-noise", type=click.Choice(["fixed", "calibrated", "gain", "floor-gain"], case_sensitive=False), default="fixed", show_default=True, help="How --plate-fit weights the points K is fitted to, with or without --plate-screen-z. fixed uses y_err as built (bg_noise floor plus any --noise-gain/--noise-alpha). calibrated estimates gain and alpha per label from the fit's own residuals and refits under them; it describes the residuals better and fits K worse, so it is not the default. gain weights by floor^2 + gain*yhat with alpha 0, the floor held and the gain calibrated from the dof-corrected residuals between refits until it settles; floor-gain calibrates the floor too. Both write plate_lm_noise_history.csv.")  # fmt: skip
@click.option("--plate-screen-noise", type=click.Choice(["calibrated", "fixed"], case_sensitive=False), default="calibrated", show_default=True, help="The ruler --plate-screen-z judges points on. calibrated fits gain and alpha to the screening pass's own residuals so bright and dim points are judged alike; fixed judges on y_err as built. Separate from --plate-noise, which sets the weights K is then fitted with.")  # fmt: skip
@click.option("--plate-screen-z", type=float, default=None, help="For --plate-fit: drop points whose |z| exceeds this and refit. The ruler is --plate-screen-noise (calibrated by default, so a dim point and a bright one are judged on the same scale); the refit uses --plate-noise's weights. 3.0 is the value measured to help; 2.5 is harmful.")  # fmt: skip
@click.option("--plate-screen-frac", type=float, default=None, help="For --plate-fit: also drop 400 nm points whose |y-yhat|/yhat exceeds this, sparing any the 485 nm channel moves with. A z-score fails at both ends of a titration -- sigma tracks the signal while model error tracks the curve -- so a 5% miss at the dim end reads as 3.6 sigma while a 36% miss at the bright end reads as 2.7. 0.12 is where reviewer calls separate; unset leaves the z-screen alone.")  # fmt: skip
@click.option("--mcmc-robust/--no-mcmc-robust", "mcmc_robust", default=False, show_default=True, help="Use a robust likelihood for --mcmc instead of a Normal. Student-t nu=3 was the best-calibrated arm on this campaign's plates.")  # fmt: skip
@click.option("--mcmc-robust-likelihood", type=click.Choice(["student_t", "mixture"], case_sensitive=False), default="student_t", show_default=True, help="Which robust likelihood --mcmc-robust selects: a heavy-tailed student_t, or a Normal/outlier contamination mixture that models the outliers rather than down-weighting them.")  # fmt: skip
@click.option("--student-t-nu", default=3.0, show_default=True, type=float, help="Student-t degrees of freedom for --mcmc-robust. Lower is heavier-tailed; pass 0 to infer nu (support above 2).")  # fmt: skip
@click.option("--mcmc-x-start-between-learn", "learn_x_start_between", is_flag=True, default=False, help="For --mcmc-x-error per_well: estimate how far apart the wells' pH axes sit (x_start_between) instead of pinning it, with --mcmc-x-start-between as the prior scale. A well's pH offset shifts its K, so pinning that scale asserts how far two wells' K may sit apart.")  # fmt: skip
@click.option("--ctr-sigma-w", "ctr_sigma_w", type=float, default=None, help="For --mcmc multi on a pH titration: let each control replicate keep its own K a learned distance from its group's, with this prior SD (pH) on that distance (K_sigma_w). Between --ctr-shared-k, which asserts the replicates agree exactly, and --ctr-free-k, which says nothing about the group; ~0.08 is what the plates show. Unset leaves the model as it was.")  # fmt: skip
@click.option("--acid-scale", is_flag=True, help="pH titrations: give each well a free factor on its most acidic step, shared by both labels, in the global lm/huber fit and in --mcmc multi. Both channels lose emission at the last acid addition by a mutant-dependent amount (the 400 nm 'turnover'); the factor frees that step's brightness but keeps its label ratio, which still informs K. On the 11 final library plates it improved bench accuracy over the plain fit and over dropping the step (arslanbaeva's scripts/score_acid_loss.py).")  # fmt: skip
@click.option("--ctr-free-k/--ctr-shared-k", "ctr_free_k", default=False, show_default=True, help="For --mcmc multi and --plate-fit: fit every well its own K rather than pooling each control group onto a shared one. Pooling buys no accuracy at the construct level and narrows the stated interval, and library wells have no group to pool with.")  # fmt: skip
@click.option("--mcmc-x-error", type=click.Choice(["deterministic", "per_well"], case_sensitive=False), default="deterministic", show_default=True, help="Latent pH axis for --mcmc multi. deterministic is one pipetting walk shared by every well; per_well gives each well its own walk, with step SDs from the measured pH errors (read noise plus accumulated pipetting). pH is measured in a few wells and their spread grows along the titration, so only per_well carries an unmeasured well's pH uncertainty into its K.")  # fmt: skip
@click.option("--mcmc-x-start-between", type=float, default=None, help="For --mcmc-x-error per_well: prior SD of each well's pH offset at the first step. It passes straight into K's interval, so set it to the measured well-to-well spread at the first step. Unset keeps the library default.")  # fmt: skip
@click.option("--mcmc-tune", default=None, type=int, help="Tuning draws per chain for --mcmc. Default is mcmc-samples // 2.")  # fmt: skip
@click.option("--mcmc-target-accept", default=None, type=float, help="NUTS target acceptance for --mcmc. Default is latent-x aware.")  # fmt: skip
@click.option("--mcmc-chains", default=None, type=click.IntRange(min=1), help="Number of NUTS chains for --mcmc. Unset keeps the sampler default (4).")  # fmt: skip
@click.option("--mcmc-seed", default=None, type=int, help="Random seed for --mcmc, making the draws reproducible. Unset leaves sampling nondeterministic.")  # fmt: skip
@click.option("--print-spec", is_flag=True, help="Print the resolved analysis specification and its signature, then exit. Two runs with the same signature fit the same model, whatever flags were typed.")  # fmt: skip
@click.option("--dry-run", is_flag=True, help="Validate inputs without processing data.")  # fmt: skip
@click.option("--detect-bad/--no-detect-bad", default=True, show_default=True, help="Run bad-well detection: discard unusable wells before fitting, writing discarded_wells.txt, and record everything atypical in atypical_wells.csv beside it.")  # fmt: skip
@click.option("--max-k-se", default=0.30, type=float, show_default=True, help="With --detect-bad, a pKa whose standard error exceeds this (pH) is undetermined: left off the K plot, marked on its figure, listed in discarded_wells.txt. Every ffit*.csv carries an 'undetermined' column regardless. Chloride ignores it: a Kd is undetermined when its SE exceeds it, and 'does not bind' (a no_binding column, listed and left off the plot likewise) when its 94% lower bound is above the highest concentration titrated.")  # fmt: skip
@click.option("--mask-outliers/--no-mask-outliers", default=False, show_default=True, help="Mask geometric point outliers before fitting.")  # fmt: skip
@click.option("--outlier-threshold", default=0.2, type=float, show_default=True, help="Threshold for geometric point outlier scoring (0-1).")  # fmt: skip
def tecan(  # ruff: ignore[complex-structure, too-many-branches, too-many-arguments, too-many-statements]
    ctx: Context,  # Click context object.
    list_file: str,
    cl: float,
    bg: bool,
    bg_adj: bool,
    nrm: bool,
    bg_mth: str,
    raw_dir: str | None,
    label: tuple[str, ...],
    sch: str | None,
    add: str | None,
    comb: bool,
    lim: tuple[float, float] | None,
    title: str,
    fit: bool,
    png: bool,
    fit_method: str,
    fit_noise: str,
    outlier: str | None,
    mcmc: str,
    nuts_sampler: str,
    mcmc_samples: int,
    noise_alpha: tuple[float, ...],
    noise_gain: tuple[float, ...],
    noise_floor: tuple[float, ...],
    noise_floor_ref_gain: tuple[float, ...],
    mcmc_noise: str,
    per_well_ye_mags: bool | None,
    noise_ye_mag: bool,
    ye_mag_parameterization: str,
    noise_mode: str,
    noise_floor_mode: str | None,
    noise_gain_mode: str | None,
    noise_alpha_mode: str | None,
    plate_fit: str | None,
    plate_noise: str,
    plate_screen_noise: str,
    plate_screen_z: float | None,
    plate_screen_frac: float | None,
    mcmc_robust: bool,
    mcmc_robust_likelihood: str,
    student_t_nu: float,
    ctr_free_k: bool,
    ctr_sigma_w: float | None,
    learn_x_start_between: bool,
    mcmc_x_error: str,
    acid_scale: bool,
    mcmc_x_start_between: float | None,
    mcmc_tune: int | None,
    mcmc_target_accept: float | None,
    mcmc_chains: int | None,
    mcmc_seed: int | None,
    dry_run: bool,
    print_spec: bool,
    detect_bad: bool,
    max_k_se: float,
    mask_outliers: bool,
    outlier_threshold: float,
) -> None:
    """Convert a list of Tecan-exported excel files into titrations.

    LIST_FILE : Path to file containing Tecan files and concentration values.

    Saves titrations as .dat files and fits all wells using 2 labels. The
    function produces:

    - K plot

    - csv tables for all labelblocks and global fittings.

    Buffer is always subtracted if scheme indicates buffer well positions.
    """
    out = ctx.obj.get("OUT", __tecan_out_dir__)
    verbose = ctx.obj.get("VERBOSE", 0)
    quiet = ctx.obj.get("QUIET", 0)
    configure_logging(verbose=verbose, quiet=quiet, log_file="ppr_tecan_cli.log")
    logger = logging.getLogger("clophfit.cli.ppr_tecan")
    logger.debug("CLI started")
    out_fp = Path(out) / "Cl" if cl else Path(out) / "pH"
    out_fp.mkdir(parents=True, exist_ok=True)
    # Derived flags: --bg-adj implies --bg; --add implies dilution correction
    # Choosing how to compute the buffer means wanting it subtracted. Without
    # this, `--bg-mth fit` silently selects a background that is then never
    # used, and the only clue is the output folder losing its `_bg` suffix.
    bg = bg or bg_adj or bg_mth != _DEFAULT_BG_MTH
    dil = add is not None
    # Options validation with clear error messages
    try:
        _validate_tecan_options(cl, bg, dil, add, sch, comb)
    except (DataValidationError, MissingDependencyError) as e:
        raise click.ClickException(str(e)) from e
    # A knob that is accepted and then ignored is worse than an error: the run
    # looks configured and is not.
    if mcmc_x_start_between is not None and mcmc_x_error.lower() != "per_well":
        msg = "--mcmc-x-start-between only applies with --mcmc-x-error per_well."
        raise click.UsageError(msg)
    # Checked here, not by FloatRange(min_open=True): the installed
    # types-click 7.1.8 stubs predate min_open and fail strict mypy.
    if max_k_se <= 0:
        msg = "must be positive; a limit of 0 calls every well undetermined."
        raise click.BadParameter(msg, param_hint="--max-k-se")
    if fit_noise.lower() != "fixed" and fit_method.lower() not in {"lm", "huber"}:
        msg = (
            f"--fit-noise {fit_noise} calibrates the lm or huber fit, not {fit_method}."
        )
        raise click.UsageError(msg)
    if (
        plate_noise.lower() in {"gain", "floor-gain"}
        and (plate_fit or "").lower() != "lm"
    ):
        msg = f"--plate-noise {plate_noise} needs --plate-fit lm."
        raise click.UsageError(msg)

    # Dry run mode: validate inputs and exit
    if dry_run:
        click.echo("🔍 Dry run mode: Validating inputs...\n")
        _dry_run_validation(_DryRunInputs(list_file, sch, add, cl, out_fp, raw_dir))
        click.echo("\n✅ Validation successful! All inputs are valid.")
        click.echo("   Remove --dry-run flag to process data.")
        return

    # Config
    tecan_config = TecanConfig(
        out_fp,
        comb,
        lim,
        title,
        fit,
        png,
        detect_bad,
        ctr_free_k,
        plate_screen_z,
        plate_screen_frac,
        plate_noise.lower(),
        plate_screen_noise.lower(),
        max_k_se=max_k_se,
        fit_noise=fit_noise.lower(),
    )

    # Load titration with error handling
    list_fp = Path(list_file)
    logger.info("Titration list: %s", list_fp.resolve())
    logger.info("%s", tecan_config)

    try:
        tit = Titration.fromlistfile(list_fp, is_ph=not cl, base_dir=raw_dir)
    except FileNotFoundError as e:
        # LIST_FILE existence is enforced by click, so this is a listed .xls.
        msg = (
            f"Tecan file listed in {list_fp} not found: {e}\n"
            f"Files are looked up in {raw_dir or list_fp.parent}; "
            f"use --raw-dir to point at the folder holding the .xls files."
        )
        raise click.ClickException(msg) from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        msg = (
            f"Error parsing list file: {list_fp}\n"
            f"Expected format: CSV with columns for file paths and concentrations.\n"
            f"Details: {e}"
        )
        raise click.ClickException(msg) from e
    except Exception as e:
        msg = (
            f"Error loading titration data from {list_fp}: {e}\n"
            f"Please check the file format and contents."
        )
        raise click.ClickException(msg) from e
    if label:
        missing = set(label) - set(tit.labelblocksgroups)
        if missing:
            msg = (
                f"No label {sorted(missing)} in {list_fp}; "
                f"it carries {sorted(tit.labelblocksgroups)}."
            )
            raise click.ClickException(msg)
        for name in list(tit.labelblocksgroups):
            if name not in label:
                del tit.labelblocksgroups[name]

    if print_spec:
        _echo_spec({
            "label": tuple(label) or "all",
            "bg": bg,
            "bg_adj": bg_adj,
            "bg_mth": bg_mth,
            "nrm": nrm,
            "dil": dil,
            "fit_method": fit_method,
            "fit_noise": fit_noise,
            "outlier": outlier,
            "mask_outliers": mask_outliers,
            "outlier_threshold": outlier_threshold,
            "detect_bad": detect_bad,
            "mcmc": mcmc,
            "mcmc_samples": mcmc_samples,
            "nuts_sampler": nuts_sampler,
            "mcmc_noise": mcmc_noise,
            "noise_mode": noise_mode,
            "floor_mode": noise_floor_mode,
            "gain_mode": noise_gain_mode,
            "alpha_mode": noise_alpha_mode,
            "noise_alpha": tuple(noise_alpha),
            "noise_gain": tuple(noise_gain),
            "noise_floor": tuple(noise_floor),
            "noise_floor_ref_gain": tuple(noise_floor_ref_gain),
            "per_well_ye_mags": per_well_ye_mags,
            "noise_ye_mag": noise_ye_mag,
            "ye_mag_parameterization": ye_mag_parameterization,
            "mcmc_robust": mcmc_robust,
            "student_t_nu": student_t_nu if mcmc_robust else None,
            "ctr_free_k": ctr_free_k,
            "acid_scale": acid_scale,
            "ctr_sigma_w": ctr_sigma_w,
            "learn_x_start_between": learn_x_start_between,
            "x_error_model": mcmc_x_error.lower(),
            "x_start_between_sigma": mcmc_x_start_between,
            "mcmc_tune": mcmc_tune,
            "mcmc_target_accept": mcmc_target_accept,
            "mcmc_chains": mcmc_chains,
            "mcmc_seed": mcmc_seed,
            "plate_fit": plate_fit,
        })
        return

    tit.params.bg = bg
    tit.params.bg_adj = bg_adj
    tit.params.dil = dil
    tit.params.nrm = nrm
    tit.params.bg_mth = bg_mth
    tit.params.fit_method = fit_method
    tit.params.outlier = outlier
    tit.params.noise_alpha = noise_alpha
    tit.params.noise_gain = noise_gain
    tit.params.noise_floor = noise_floor
    tit.params.noise_floor_ref_gain = noise_floor_ref_gain
    tit.params.mask_outliers = mask_outliers
    tit.params.acid_scale = acid_scale
    tit.params.outlier_threshold = outlier_threshold
    logger.info("%s", tit.params)

    # Load additions file with error handling
    if add:
        try:
            tit.load_additions(Path(add))
            logger.info("Additions: %s", tit.additions)
        except FileNotFoundError:
            msg = (
                f"Additions file not found: {add}\n"
                f"This file is required when using --cl option."
            )
            raise click.ClickException(msg) from None
        except Exception as e:
            msg = (
                f"Error loading additions file {add}: {e}\n"
                f"Expected format: Initial volume and addition volumes."
            )
            raise click.ClickException(msg) from e

    if cl and tit.additions:
        try:
            tit.x = calculate_conc(tit.additions, cl)
            logger.info("%s", tit.x)
        except Exception as e:
            msg = (
                f"Error calculating chloride concentrations: {e}\n"
                f"Please check additions file format and --cl value."
            )
            raise click.ClickException(msg) from e

    # Load scheme file with error handling
    if sch:
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            tit.load_scheme(Path(sch))
            f = tit.buffer.plot(title=title)
            f.savefig(out_fp / "buffer.png")
            f = tit.buffer.plot(nrm=True, title=title)
            f.savefig(out_fp / "buffer_norm.png")
            logger.info("%s", tit.scheme)
        except FileNotFoundError:
            msg = (
                f"Scheme file not found: {sch}\n"
                f"This file is required when using --bg option."
            )
            raise click.ClickException(msg) from None
        except Exception as e:
            msg = (
                f"Error loading scheme file {sch}: {e}\n"
                f"Expected format: Tab-separated file with well positions and sample names."
            )
            raise click.ClickException(msg) from e

    # Export metadata and plots with error handling
    try:
        with (out_fp / "metadata-labels.txt").open("w", encoding="utf-8") as fp:
            for lbg in tit.labelblocksgroups.values():
                pprint.pprint(lbg.metadata, stream=fp)
        f = tit.plot_temperature(title=title)
        f.savefig(out_fp / "temperatures.png")
    except PermissionError as e:
        msg = (
            f"Permission denied writing to output directory: {out_fp}\n"
            f"Please check directory permissions."
        )
        raise click.ClickException(msg) from e
    except OSError as e:
        msg = f"Error writing output files to {out_fp}: {e}"
        raise click.ClickException(msg) from e

    # Output and export with error handling
    if acid_scale and mcmc in {"single", "single-refit"}:
        msg = "--acid-scale is implemented for the global lm/huber fit and --mcmc multi only."
        raise click.UsageError(msg)
    mcmc_spec = (
        None
        if mcmc == "None"
        else McmcSpec(
            model=cast('Literal["single", "single-refit", "multi"]', mcmc),
            sampler=SamplerConfig(
                n_samples=mcmc_samples,
                nuts_sampler=nuts_sampler,
                n_tune=mcmc_tune,
                target_accept=mcmc_target_accept,
                chains=mcmc_chains,
                random_seed=mcmc_seed,
            ),
            # nu=0 is the CLI's way of asking for an inferred nu, which the
            # library spells as None.
            robust=RobustConfig(
                enabled=mcmc_robust,
                likelihood=cast(
                    'Literal["student_t", "mixture"]', mcmc_robust_likelihood
                ),
                nu=student_t_nu if student_t_nu > 0 else None,
            ),
            ctr_free_k=ctr_free_k,
            ctr_sigma_w_prior=ctr_sigma_w,
            acid_scale=acid_scale,
            learn_x_start_between=learn_x_start_between,
            structured_noise=mcmc_noise == "structured",
            per_well_ye_mags=per_well_ye_mags,
            noise_ye_mag=noise_ye_mag,
            ye_mag_parameterization=cast(
                'Literal["centered", "hierarchical", "separable", "separable_step"]',
                ye_mag_parameterization,
            ),
            noise_mode=cast('Literal["centered", "fixed"]', noise_mode),
            floor_mode=cast('Literal["centered", "fixed"] | None', noise_floor_mode),
            gain_mode=cast('Literal["centered", "fixed"] | None', noise_gain_mode),
            alpha_mode=cast('Literal["centered", "fixed"] | None', noise_alpha_mode),
            x_error_model=cast(
                'Literal["deterministic", "per_well"]', mcmc_x_error.lower()
            ),
            x_start_between_sigma=mcmc_x_start_between,
        )
    )
    logger.info("mcmc: %s", mcmc_spec)
    try:
        export_data_fit(tit, tecan_config, mcmc_spec, plate_fit)
    except Exception as e:
        msg = (
            f"Error during data export and fitting: {e}\n"
            f"Check the log file for more details: ppr_tecan_cli.log"
        )
        raise click.ClickException(msg) from e


def _validate_tecan_options(  # ruff: ignore[too-many-arguments]
    cl: float | None,
    bg: bool,
    dil: bool,
    add: str | None,
    sch: str | None,
    comb: bool,
) -> None:
    """Validate tecan command options.

    Parameters
    ----------
    cl : float | None
        Chloride concentration option.
    bg : bool
        Background subtraction flag.
    dil : bool
        Dilution correction flag.
    add : str | None
        Additions file path.
    sch : str | None
        Scheme file path.
    comb : bool
        All combinations flag.

    Raises
    ------
    MissingDependencyError
        If required files are not specified.
    DataValidationError
        If option combinations are invalid.
    """
    if cl and not add:
        raise MissingDependencyError(
            missing_file="additions file (--add)",
            required_by="--cl option",
            reason="Chloride titrations require addition volumes to calculate concentrations.",
        )
    if bg and not sch:
        raise MissingDependencyError(
            missing_file="scheme file (--sch)",
            required_by="--bg option",
            reason="Buffer subtraction requires a plate scheme to identify buffer wells.",
        )
    if comb and not (bg and sch and dil):
        msg = "All combinations mode requires --bg, --sch, and --add to be specified."
        raise DataValidationError(
            msg,
            suggestions=[
                "Add --bg --sch scheme.txt --add additions.pH flags",
                "Or remove --all flag if you don't need all combinations",
            ],
        )


_MAX_MISSING_FILES_PREVIEW = 3


class _DryRunInputs(NamedTuple):
    """Grouped CLI inputs validated by :func:`_dry_run_validation`."""

    list_file: str
    sch: str | None
    add: str | None
    cl: float | None
    out_fp: Path
    raw_dir: str | None = None


def _dry_run_validation(inputs: _DryRunInputs) -> None:
    """Perform dry-run validation of input files.

    Parameters
    ----------
    inputs : _DryRunInputs
        Grouped list, scheme, additions, chloride, output, and raw-dir inputs.

    Raises
    ------
    click.ClickException
        If validation fails.
    """
    list_file, sch, add, cl, out_fp, raw_dir = inputs
    list_fp = Path(list_file)

    # Validate list file
    click.echo(f"✓ List file exists: {list_fp}")
    try:
        df = pd.read_csv(list_fp, names=["filenames", "x", "x_err"])
        click.echo(f"  - Contains {len(df)} entries")
        if len(df) == 0:
            msg = "List file is empty"
            raise click.ClickException(msg)  # ruff: ignore[raise-within-try] #FIXME:
    except click.ClickException:
        raise
    except Exception as e:
        msg = f"Error reading list file: {e}"
        raise click.ClickException(msg) from e

    # Validate the Tecan files the list file points to
    root = Path(raw_dir) if raw_dir else list_fp.parent
    missing = [f for f in df["filenames"] if not (root / f).is_file()]
    if missing:
        msg = (
            f"{len(missing)} of {len(df)} Tecan files not found in {root}: "
            f"{', '.join(map(str, missing[:_MAX_MISSING_FILES_PREVIEW]))}"
            f"{' ...' if len(missing) > _MAX_MISSING_FILES_PREVIEW else ''}\n"
            f"Use --raw-dir to point at the folder holding the .xls files."
        )
        raise click.ClickException(msg)
    click.echo(f"✓ Tecan files found: {len(df)} in {root}")

    # Validate scheme file
    if sch:
        sch_fp = Path(sch)
        click.echo(f"✓ Scheme file exists: {sch_fp}")
        try:
            sch_df = pd.read_csv(sch_fp, sep="\t")
            click.echo(f"  - Contains {len(sch_df)} well definitions")
        except Exception as e:
            msg = f"Error reading scheme file: {e}"
            raise click.ClickException(msg) from e

    # Validate additions file
    if add:
        add_fp = Path(add)
        click.echo(f"✓ Additions file exists: {add_fp}")
        try:
            with add_fp.open(encoding="utf-8") as f:
                lines = f.readlines()
            click.echo(f"  - Contains {len(lines)} lines")
            if cl:
                click.echo(f"  - Will calculate [Cl] with {cl} mM stock")
        except Exception as e:
            msg = f"Error reading additions file: {e}"
            raise click.ClickException(msg) from e

    # Validate output directory
    click.echo(f"✓ Output directory: {out_fp}")
    if not out_fp.exists():
        click.echo(f"  - Will create: {out_fp}")
    else:
        click.echo("  - Already exists (files may be overwritten)")


########################################
# pr.enspire                           #
########################################
@ppr.command()
@click.pass_context
@click.argument("csv_f", type=cPath(exists=True, path_type=str))
@click.argument("note_f", type=cPath(exists=True), required=False)
@click.option(
    "-b",
    "bands",
    multiple=True,
    default=None,
    nargs=3,
    type=(str, int, int),
    help="Label and band interval (format: LABEL LOWER UPPER)",
)
@click.option("--method", type=click.Choice(["band", "svd", "both", "global", "global-bayes", "global-bayes-marginal", "vanthoff", "vanthoff-bayes", "bayes", "all"], case_sensitive=False), default="band", show_default=True, help="How spectra become one number per well. band averages named windows (anionic 480-495, neutral 395-410, emission 500-520) and lets them share K; svd projects whole spectra on their first principal component, as this command always did. Neither is uniformly more precise -- on replicate rows of one plate they trade places -- but band keeps plateaus in measured units, cross-checks its bands against each other, and names a readout that has stopped titrating. global (prototype) fits every wavelength of the excitation scan with one K, the two species spectra solved linearly and a per-well amplitude, with a jackknife error; it writes <stem>_K_global.csv and one <stem>_<sample>_<temp>_panels.pdf per group, plus <stem>_global_inspect.txt with --inspect. (The discarded wells stay in --global-qc and the K table, not the panel PDF.) global-bayes puts the same whole-spectrum model on a seeded NUTS posterior with a latent per-well titrant axis (--x-sigma required), reporting the K median and 94 % HDI in <stem>_K_global_bayes.csv, the latent axis in <stem>_x_global_bayes.csv and <stem>_global_bayes.pdf.  vanthoff fits every temperature of the note at once, tied by pKa(T) = dH/(2.303 R T) - dS/(2.303 R): the bands of all temperatures share one thermodynamic pair while plateaus stay free per band and temperature; pH plates fit pKa(T) (linear in 1/T), ligand plates pKd(T) - so Kd is exponential in 1/T and reported in the x units; needs at least two temperatures and writes <stem>_K_vanthoff.csv and <stem>_K_vanthoff.pdf (curves with the joint model, and the constant vs 1/T). vanthoff-bayes puts that same joint model on a seeded NUTS posterior: the shared thermodynamic pair and the per-band plateaus are sampled together, so dH and dS carry their own 94 % HDI instead of a propagated one; it writes <stem>_K_vanthoff_bayes.csv (K per temperature with hdi03/hdi97, r_hat and ess_bulk), <stem>_K_vanthoff_bayes.pdf (the joint model with its credible band, and the constant vs 1/T) and one <stem>_<sample>_vanthoff_bayes.nc trace per sample; --x-sigma adds the latent per-well titrant axis, as for the other Bayesian methods. bayes puts a seeded PyMC/NUTS posterior on the same corrected band readouts as band, combining the excitation and emission scans into one shared K, and reports the median and 94 % HDI (hdi03/hdi97, r_hat, ess_bulk) in <stem>_K_bayes.csv and <stem>_K_bayes.pdf: the successor of the legacy svd+emcee HDI - same Bayesian interval, stable readout, reproducible seed. both runs band and svd; all runs band, svd and global.")  # fmt: skip
@click.option("--join", "joins", multiple=True, metavar="CSV:NOTE", help="Additional export:note pair to fit together with the first, for a titration split over several EnSpire files (e.g. 20 C in one export, 37 C in another, each with its own note). The exports must carry disjoint measurement labels; the notes are concatenated (each keeps its Temp) and written next to the outputs as <stem>_joined_note.csv, so nothing under data/ changes. --method vanthoff then fits one thermodynamic pair across every temperature.")  # fmt: skip
@click.option("--export", "export", type=click.Choice(["auto", "always", "never"], case_sensitive=False), default="auto", show_default=True, help="What to do with the per-measurement tables and figures written before the fit. auto (default) writes the canonical <export>_<label>.csv/json/png and rewrites one only when the export is newer than it, so calling several methods on one plate reuses the same files instead of piling up timestamped copies; always rewrites everything; never writes nothing and only fits.")  # fmt: skip
@click.option("--buffer/--no-buffer", default=True, show_default=True, help="For --method band: subtract the buffer well of the same plate column. A row sitting at the instrument's floor is treated as buffer even when the note names it after the mutant.")  # fmt: skip
@click.option("--screen", default=bands_module.DEFAULT_SCREEN, show_default=True, help="Point screen for --method band, as method:threshold:min_keep. 'studentized' takes a family-wise alpha and a Bonferroni-corrected Student-t cutoff; 'mad' takes a robust z. Pass 'none' to fit every point.")  # fmt: skip
@click.option("--min-rho", default=0.8, show_default=True, help="For --method band: smallest |Spearman rho| between a band and the titrant, taken within each replicate row, for that band to carry K. A flat band has no K to give.")  # fmt: skip
@click.option("--band-name", "band_names", multiple=True, type=click.Choice(["exc_anionic", "exc_neutral", "em_exc420", "em_exc278"]), help="Bands allowed into the shared K; repeatable. Default is the two direct readouts (exc_anionic + em_exc420); exc_neutral and em_exc278 are reported but not fitted by default.")  # fmt: skip
@click.option("--scan", "scans", multiple=True, type=click.Choice(["exc_anionic", "exc_neutral", "em_exc420", "em_exc278"]), help="Scans --method global takes whole; repeatable. Default is the excitation scan only: on the 25-plate readout sweep, adding the 420 nm emission scan whole widened the between-session spread (see clophfit/prenspire/spectral.py), so all-scans is offered for comparison, not as the default.")  # fmt: skip
@click.option("--seed", type=int, default=0, show_default=True, help="NUTS seed for --method bayes; fixes the posterior draws so the 94 % HDI is reproducible run to run.")  # fmt: skip
@click.option("--draws", type=int, default=4000, show_default=True, help="Posterior draws per chain for the Bayesian methods (tuning is half of that). The band-level ones sample in under a minute; each whole-spectrum one takes proportionally longer per draw, so budget minutes-per-scan there.")  # fmt: skip
@click.option("--chains", type=click.IntRange(min=1), default=4, show_default=True, help="MCMC chains for the Bayesian methods. Four chains are the sanity floor for R-hat: a lost or label-flipped chain is cheap insurance when it can be seen, and a fixed --seed stays reproducible whether the chains run sequentially or in parallel.")  # fmt: skip
@click.option("--cores", type=click.IntRange(min=1), help="CPU cores for --method bayes chain parallelism. Unset runs the chains in parallel (PyMC default, capped at the core count); 1 runs them sequentially.")  # fmt: skip
@click.option("--target-accept", type=float, help="NUTS target acceptance for --method bayes, in (0, 1). Unset keeps the sampler default; higher (e.g. 0.9) trades step size for better mixing.")  # fmt: skip
@click.option("--acid-state", is_flag=True, help="For --method global-bayes and global-bayes-marginal (pH plates): add the third acid state the two-state model cannot represent - the recurrent signal drop of the most acidic well that stays a straight line in the label ratio. Same conventions as --method global (band-level acid_scale): the acid state gets its own pK below K and its own plateau, E2_; the circle of acidic wells the user notes across many EnSpire plates is the case in point.")  # fmt: skip
@click.option("--x-sigma", type=str, default=None, help="For --method bayes, global-bayes and vanthoff-bayes: prior SD on each well's true axis value - x_true ~ N(note value, sigma), per well, in the axis's own units (pH, or mM on ligand plates). The note value is an instrument reading, not an exact level, so a latent axis is a defensible option: 'auto' pools the note's replicate-row spread per index column (A04/B04/C04 are the same nominal target) into the sigma, a number (e.g. 0.15) fixes it. Default: the axis is fixed at the note values; the whole-spectrum Bayesian methods require it.")  # fmt: skip
@click.option("--robust/--no-robust", "robust", default=True, show_default=True, help="For --method bayes: Student-t (nu = 3) likelihood, which down-weights outliers; --no-robust gives a plain Normal, i.e. maximum likelihood with the latent axis still on.")  # fmt: skip
@click.option("--bayes-screen", type=str, default=None, help="For --method bayes: point screen applied to the readouts before sampling, in the --screen form (e.g. studentized:0.05:5). A well flagged in any band is dropped from every band, so the posterior and the band fit see the same wells and the latent axis stays shared. Off by default: --robust instead down-weights outliers, which is not the same as deleting them.")  # fmt: skip
@click.option("--global-screen", type=float, default=None, help="For --method global: family-wise level (e.g. 0.05) of a well screen - a well whose standardized whole-spectrum residual is beyond the chi-square cutoff is dropped and the fit repeated. Off by default, because this method never screened; set it when comparing against the screened band path.")  # fmt: skip
@click.option("--global-boot", type=int, default=0, show_default=True, help="For --method global: well-cluster bootstrap replicates for the reported interval. The default 0 reports the delete-one-well jackknife only. Every replicate resamples whole wells - all scans and all wavelengths of a well together - because the wavelengths of a spectrum are not independent observations; the Jacobian error of a whole-spectrum fit comes out orders of magnitude too small. 500 is a sensible choice.")  # fmt: skip
@click.option("--global-qc", type=float, default=None, metavar="Z", help="For --method global: run the per-scan screen (leave-one-out rank-2 SVD of each corrected scan) before the fit and drop wells whose residual exceeds median + Z*MAD in any scan, from every scan of the group. Z=6 is what the pH verification plates were calibrated at. Wells whose peak is below 5 %% of the plate median are never tested, so the dark saturating end of a ligand ladder is not mistaken for a broken well. The dropped wells are recorded in the K table.")  # fmt: skip
@click.option("--global-strata", type=click.Choice(["none", "column"], case_sensitive=False), default="none", show_default=True, help="For --method global --global-boot: 'column' resamples wells within each plate column, which keeps every replicate spanning the titration axis (the design-preserving choice for these plates); 'none' resamples from one pool.")  # fmt: skip
@click.option("--normalise-isosbestic", "isosbestic", type=str, default=None, metavar="[NM|auto|auto-scale]", help="Divide every well by its signal at the isosbestic point before fitting, which removes how much protein the well holds, its path length and its brightness while keeping the titration's shape. 'auto' locates one crossing per (sample, temperature). 'auto-scale' keeps those crossings and puts a sample's temperatures on ONE amplitude scale, which is what preserves the measured 20/37 C brightness ratio - use it whenever the two temperatures are drawn or fitted together. A number is a fixed wavelength in nm. A chloride titration has no such point - bound and free differ in brightness without crossing - so the auto forms normalise nothing there and change no number. Writes <stem>_isosbestic.csv recording the wavelength each group used.")  # fmt: skip
@click.option(
    "--iso-window",
    "iso_band",
    default=None,
    show_default="3",
    type=click.FloatRange(min=0.0),
    metavar="NM",
    help="Width in nm of the band averaged at the isosbestic crossing to form the "
    "per-well reference, for --normalise-isosbestic. It does nothing without "
    "that option and is refused there rather than ignored. The reference is "
    "always a mean over the band, never a sum, and every measured point inside "
    "it is used: 3 nm averages 3 points of a 1 nm grid and 2 points of a 2 nm "
    "grid, and 0 means the interpolated crossing itself. It changes only a "
    "per-well constant, which the global fit's free well amplitude already "
    "fits, so it is a readout-noise choice and moves K far less than its error.",
)
@click.option(
    "--species-min",
    "species_min",
    default=None,
    type=float,
    metavar="FLOOR",
    help="Bound every species spectrum below by FLOOR, per wavelength, for the "
    "whole-spectrum methods. 0 is the physical choice for fluorescence - a pure "
    "species cannot emit negative counts - and it removes the negative lobe the "
    "unconstrained solve gives a species that should be dark in a scan (L8 A08 "
    "emission 420: -8 percent of its own peak at 499 nm). A small negative floor "
    "keeps buffer-subtraction noise unclipped. With --method global the bound is "
    "inside the fit, so K moves slightly and --global-qc still screens the "
    "unconstrained model; with global-bayes and global-bayes-marginal it can only "
    "censor the reported species draws, leaving the K posterior untouched.",
)
@click.option(
    "--ppc",
    "ppc",
    is_flag=True,
    default=False,
    help="For --method global-bayes: also write <stem>_ppc.csv, a posterior "
    "predictive check. Each scan is simulated from a sample of posterior draws and "
    "the observed residual RMS and its lag-1 autocorrelation along wavelength are "
    "compared with the same statistics under the replicates, as two-sided Bayesian "
    "p-values. RMS far from the replicates means the noise scale is wrong; lag-1 "
    "far above them means the model's independent per-wavelength errors cannot "
    "describe the residual structure.",
)
@click.option(
    "--dpH-buffer",
    "dph_buffer",
    default=None,
    metavar="[auto|DPH]",
    help="For --method vanthoff-bayes: tie every temperature's latent axis to the "
    "lowest one, well by column, plus that column's pH difference. 'auto' reads the "
    "difference from the note, which is what the buffer actually did; a number uses "
    "it for every column. A column is one nominal target and the buffer's pH moves "
    "with temperature the same way across a plate, so only the reference "
    "temperature's axes stay free - which removes the between-temperature half of "
    "the K-x ridge. Ignored when the wells do not pair by column.",
)
@click.option(
    "--x-pool",
    "x_pool",
    type=click.Choice(["free", "shared", "partial"], case_sensitive=False),
    default="free",
    show_default=True,
    help="How the Bayesian methods treat the latent titrant axis across replicate "
    "wells. free: every well gets its own x_true ~ N(x_note, sigma), which is right "
    "where the recorded axis can be wrong (an unverified Cl concentration) but "
    "makes three wells of one solution three different truths. shared: the wells "
    "of one replicate group share a single truth. partial: they share a group "
    "truth while a well may still deviate, with the deviation scale inferred - it "
    "nests the other two and reports how much pooling the data warrant. Default "
    "free: a 45-plate recovery study shows pooling cuts the K bias roughly in "
    "half (0.028 to 0.022 pH under a poisoned axis, and 0.005 to 0.0001 on a clean "
    "one) but at 4 chains x 2000 draws the pooled fits still mix poorly there "
    "(ESS ~20-70, r-hat up to 1.6) while free is clean, so it is offered rather "
    "than assumed. The shared and partial structures need --x-sigma.",
)
@click.option(
    "--x-groups",
    "x_groups",
    type=click.Choice(["column", "none"], case_sensitive=False),
    default="column",
    show_default=True,
    help="What a replicate group is for --x-pool: 'column' reads it from the "
    "trailing column number of the well label (A04/B04/C04 are one solution), as "
    "the column QC strata do; 'none' makes every well its own group.",
)
@click.option(
    "--inspect",
    "inspect",
    is_flag=True,
    default=False,
    help="With --method global, also write <stem>_global_inspect.txt: per "
    "(sample, temp) K and its errors, the wells used and the wells the screen "
    "dropped, each scan's basic/acidic species peaks, how far below zero either "
    "species dips, the 2-state residual sv, and every well's basic fraction and "
    "amplitude. Text only - the panel figure and --global-qc output are unchanged.",
)
def enspire(  # ruff: ignore[too-many-arguments, complex-structure, too-many-branches, too-many-statements]
    ctx: Context,
    csv_f: str,
    note_f: str | None,
    bands: tuple[Any],
    method: str,
    buffer: bool,
    screen: str,
    min_rho: float,
    band_names: tuple[str, ...],
    scans: tuple[str, ...],
    seed: int,
    draws: int,
    chains: int,
    cores: int | None,
    target_accept: float | None,
    x_sigma: str | None,
    acid_state: bool,
    global_screen: float | None,
    robust: bool,
    bayes_screen: str | None,
    export: str,
    joins: tuple[str, ...],
    isosbestic: str | None,
    iso_band: float | None,
    species_min: float | None,
    global_boot: int,
    global_strata: str,
    global_qc: float | None,
    inspect: bool,
    x_pool: str,
    x_groups: str,
    dph_buffer: str | None,
    ppc: bool,
) -> None:
    """Save spectra as csv tables from EnSpire xls file."""
    verbose = ctx.obj.get("VERBOSE", 0)
    out = ctx.obj.get("OUT", __enspire_out_dir__)
    exports = [EnspireFile(Path(csv_f), verbose=verbose)]
    notes = [Path(note_f)] if note_f is not None else []
    for pair in joins:
        csv_part, _, note_part = pair.rpartition(":")
        if not csv_part or not note_part:
            ctx.fail(f"--join expects CSV:NOTE, got {pair!r}")
        exports.append(EnspireFile(Path(csv_part), verbose=verbose))
        notes.append(Path(note_part))
    ef = merge_enspire_files(exports) if len(exports) > 1 else exports[0]
    written: list[str] = []
    for one in exports:
        written += one.export_measurements(Path(out), mode=export)
    note_f = str(notes[0]) if notes else ""
    if len(notes) > 1:
        # the joined note is a derived input: it lives with the outputs, not in data/
        frames = [pd.read_csv(path) for path in notes]
        joined = pd.concat(frames, ignore_index=True)
        note_fp = (
            Path(out) / f"{Path(note_f).name.removesuffix('_note.csv')}_joined_note.csv"
        )
        joined.to_csv(note_fp, index=False)
        provenance = {
            "inputs": [
                str(path)
                for path in [Path(csv_f), *(Path(j.rpartition(":")[0]) for j in joins)]
            ],
            "notes": [str(path) for path in notes],
            "labels": sorted(ef.measurements),
            "temperatures": sorted({
                str(m["metadata"].get("temp")) for m in ef.measurements.values()
            }),
        }
        note_fp.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
        note_f = str(note_fp)
        if verbose:
            print(f"joined {len(notes)} notes into {note_fp}")
    if verbose and not written:
        print(f"measurement tables already up to date in {out}; nothing rewritten")
    if not note_f:
        return
    if x_sigma is not None and method not in {
        "bayes",
        "global-bayes",
        "global-bayes-marginal",
        "vanthoff-bayes",
    }:
        ctx.fail(
            "--x-sigma applies to Bayesian methods only (it is a posterior prior, not an LM option)"
        )
    scan_methods = {"global", "global-bayes", "global-bayes-marginal", "all"}
    if scans and method not in scan_methods:
        ctx.fail(
            f"--scan selects whole scans for {sorted(scan_methods)}; "
            f"--method {method} takes its readouts from -b/--bands or from no scan at all"
        )
    qc_methods = {
        "global",
        "global-bayes",
        "global-bayes-marginal",
        "svd",
        "both",
        "all",
    }
    if global_qc is not None and method not in qc_methods:
        ctx.fail(
            f"--global-qc screens the whole-spectrum methods {sorted(qc_methods)} "
            f"(got --method {method})"
        )
    whole_ls_options = {
        "--global-boot": global_boot > 0,
        "--global-strata": global_strata.lower() != "none",
        "--global-screen": global_screen is not None,
    }
    ignored = [name for name, given in whole_ls_options.items() if given]
    if ignored and method not in {"global", "all"}:
        ctx.fail(
            f"{', '.join(ignored)} apply to --method global only "
            f"(got --method {method}); they would be silently ignored"
        )
    if inspect and method not in {"global", "all"}:
        ctx.fail(
            f"--inspect dumps the global fit (got --method {method}); "
            "it would be silently ignored"
        )
    bayes_methods = {
        "bayes",
        "global-bayes",
        "global-bayes-marginal",
        "vanthoff-bayes",
    }
    if method not in bayes_methods and (x_pool != "free" or x_groups != "column"):
        ctx.fail(
            f"--x-pool and --x-groups shape the Bayesian latent axis "
            f"{sorted(bayes_methods)} (got --method {method})"
        )
    if ppc and method != "global-bayes":
        ctx.fail(
            f"--ppc simulates the whole-spectrum free-species posterior (got "
            f"--method {method})"
        )
    if dph_buffer is not None and method != "vanthoff-bayes":
        ctx.fail(
            f"--dpH-buffer ties the axes of the joint van't Hoff fit (got "
            f"--method {method}); the per-temperature fitters have no axis to tie"
        )
    if method in bayes_methods and x_pool != "free" and x_sigma is None:
        # a pooled truth needs a scale; without one, say so and fall back rather
        # than silently fitting a different structure than the table will record
        click.echo(
            f"note: --x-pool {x_pool} needs --x-sigma (the recorded axis's error "
            "scale, 'auto' to estimate it from the replicate spread); fitting the "
            "free axis instead and recording x_pool=free",
            err=True,
        )
        x_pool = "free"
    species_min_methods = {
        "global",
        "global-bayes",
        "global-bayes-marginal",
        "all",
    }
    if species_min is not None and method not in species_min_methods:
        ctx.fail(
            f"--species-min bounds the species of the whole-spectrum fits "
            f"{sorted(species_min_methods)} (got --method {method}); it would be "
            "silently ignored"
        )
    if iso_band is not None and isosbestic is None:
        # the width only exists for the normalisation it shapes; without one it
        # used to look like a knob that did nothing, which is how it was reported
        ctx.fail(
            "--iso-window sets the band of the isosbestic reference, so it needs "
            "--normalise-isosbestic (a wavelength, 'auto' or 'auto-scale')"
        )
    window_methods = {"global", "global-bayes", "global-bayes-marginal", "all"}
    if iso_band is not None and method not in window_methods:
        ctx.fail(
            f"--iso-window shapes the whole-spectrum normalisation "
            f"{sorted(window_methods)} (got --method {method}); the band and svd "
            "paths divide by the reference whole"
        )
    if acid_state and method not in {"global-bayes", "global-bayes-marginal"}:
        ctx.fail(
            "--acid-state applies to --method global-bayes and "
            "global-bayes-marginal only"
        )
    iso = _parse_isosbestic(isosbestic)
    # one record of this invocation, written beside whatever each method produces
    run_provenance = _provenance(ctx)
    if iso is not None:
        _write_isosbestic_table(ef, Path(note_f), Path(out), isosbestic=iso)
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        if method in {"band", "both", "all"}:
            _echo_band_summaries(
                run_bands(
                    ef,
                    Path(note_f),
                    Path(out),
                    buffer=buffer,
                    screen=None if screen.lower() == "none" else screen,
                    min_rho=min_rho,
                    wanted=band_names or bands_module.DIRECT_BANDS,
                    isosbestic=iso,
                    provenance=run_provenance,
                )
            )
        if method in {"svd", "both", "all"}:
            _echo_messages(
                run_svd(
                    ef,
                    Path(note_f),
                    Path(out),
                    list(bands),
                    verbose,
                    isosbestic=iso,
                    qc=global_qc,
                    provenance=run_provenance,
                ).messages
            )
        if method in {"global", "all"}:
            _echo_global_run(
                run_global(
                    ef,
                    Path(note_f),
                    Path(out),
                    buffer=buffer,
                    scans=scans,
                    screen=global_screen,
                    isosbestic=iso,
                    iso_band=3.0 if iso_band is None else iso_band,
                    species_min=species_min,
                    n_boot=global_boot,
                    strata=global_strata,
                    qc=global_qc,
                    inspect=inspect,
                    provenance=run_provenance,
                )
            )
        if method == "vanthoff":
            _echo_messages(
                run_vanthoff(
                    ef,
                    Path(note_f),
                    Path(out),
                    buffer=buffer,
                    isosbestic=iso,
                    provenance=run_provenance,
                ).messages
            )
        if method == "vanthoff-bayes":
            _echo_messages(
                run_vanthoff_bayes(
                    ef,
                    Path(note_f),
                    Path(out),
                    buffer=buffer,
                    x_sigma=x_sigma,
                    seed=seed,
                    draws=draws,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    isosbestic=iso,
                    provenance=run_provenance,
                    dph_buffer=dph_buffer,
                ).messages
            )
        if method == "bayes":
            _echo_messages(
                run_bayes(
                    ef,
                    Path(note_f),
                    Path(out),
                    buffer=buffer,
                    band_names=band_names,
                    seed=seed,
                    draws=draws,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    x_sigma=x_sigma,
                    robust=robust,
                    screen=bayes_screen,
                    isosbestic=iso,
                    provenance=run_provenance,
                ).messages
            )
        if method == "global-bayes":
            if x_sigma is None:
                ctx.fail("--method global-bayes requires --x-sigma (auto or a pH SD)")
            _echo_messages(
                run_global_bayes(
                    ef,
                    Path(note_f),
                    Path(out),
                    fitter=fit_spectra_global_bayes,
                    suffix="global_bayes",
                    label="global Bayes",
                    buffer=buffer,
                    scans=scans,
                    x_sigma=x_sigma,
                    acid_state=acid_state,
                    seed=seed,
                    draws=draws,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    isosbestic=iso,
                    species_min=species_min,
                    iso_band=3.0 if iso_band is None else iso_band,
                    x_pool=x_pool,
                    x_groups=x_groups,
                    provenance=run_provenance,
                    ppc=ppc,
                ).messages
            )
        if method == "global-bayes-marginal":
            if x_sigma is None:
                ctx.fail(
                    "--method global-bayes-marginal requires --x-sigma (auto or a pH SD)"
                )
            _echo_messages(
                run_global_bayes(
                    ef,
                    Path(note_f),
                    Path(out),
                    fitter=fit_spectra_global_bayes_marginal,
                    suffix="global_bayes_marginal",
                    label="global Bayes marginal",
                    buffer=buffer,
                    scans=scans,
                    x_sigma=x_sigma,
                    acid_state=acid_state,
                    seed=seed,
                    draws=draws,
                    chains=chains,
                    cores=cores,
                    target_accept=target_accept,
                    isosbestic=iso,
                    species_min=species_min,
                    iso_band=3.0 if iso_band is None else iso_band,
                    x_pool=x_pool,
                    x_groups=x_groups,
                    provenance=run_provenance,
                ).messages
            )
    except EnspireInputError as err:
        raise click.UsageError(str(err)) from err
    except EnspireRunError as err:
        raise click.ClickException(str(err)) from err


def _echo_band_summaries(run: EnspireRunResult[BandSummary]) -> None:
    """Print the shared-K line for each sample of a band run."""
    for r in run.summaries:
        print(
            f"{r.sample} at {r.temp}: K {r.K:.3f} [{r.lo:.3f}, {r.hi:.3f}] "
            f"on {r.n} points, {r.dropped} screened out"
        )


def _echo_messages(messages: Sequence[EnspireMessage]) -> None:
    """Print a run's notes and warnings in the order the run recorded them."""
    for message in messages:
        click.echo(message.text)


def _echo_global_run(run: EnspireRunResult[GlobalSummary]) -> None:
    """Print a whole-spectrum run's messages, then its jackknife summary lines."""
    _echo_messages(run.messages)
    for r in run.summaries:
        print(
            f"{r.sample} at {r.temp}: K {r.K:.3f} ± {r.se:.3f} (jackknife) "
            f"on {r.n} wells, residual sv ratio {r.sv_ratio:.1f}"
        )


def _parse_isosbestic(value: str | None) -> float | str | None:
    """Turn ``--normalise-isosbestic`` into a wavelength, ``"auto"`` or None.

    Parameters
    ----------
    value : str | None
        The option as typed: unset, ``"auto"`` (one crossing per sample and
        temperature), ``"auto-scale"`` (the same crossings, with the amplitude
        shared across a sample's temperatures), or a wavelength in nm.

    Returns
    -------
    float | str | None
        ``None`` when the option is off, ``"auto"`` or ``"auto-scale"``, or the
        wavelength.

    Raises
    ------
    click.BadParameter
        If the value is neither ``"auto"`` nor a positive wavelength.
    """
    if value is None:
        return None
    text = value.strip().lower()
    modes = {
        "auto": "auto",
        "yes": "auto",
        "true": "auto",
        "auto-scale": "auto-scale",
        "scale": "auto-scale",
        "no": None,
        "none": None,
        "false": None,
        "": None,
    }
    if text in modes:
        return modes[text]
    try:
        lam = float(text)
    except ValueError as err:
        msg = f"--normalise-isosbestic expects a wavelength in nm or 'auto', got {value!r}"
        raise click.BadParameter(msg) from err
    if lam <= 0:
        msg = f"--normalise-isosbestic expects a positive wavelength, got {lam}"
        raise click.BadParameter(msg)
    return lam


def _provenance(ctx: Context) -> dict[str, object]:
    """Collect the parameters one invocation was given, ready to be written as JSON.

    ``ctx.params`` holds what click parsed, so the record is the invocation rather
    than a hand-written summary that can drift from it. Unset options are left
    out, keys are sorted by the writer, and nothing here is time-dependent: the
    same command produces the same file.
    """
    out: dict[str, object] = {"command": ctx.info_name or "enspire"}
    for key, value in ctx.params.items():
        if value is None or value in ((), []):
            continue
        if isinstance(value, tuple):
            out[key] = [str(item) for item in value]
        elif isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _write_isosbestic_table(
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    *,
    isosbestic: float | str,
) -> pd.DataFrame:
    """Record the crossing each (sample, temperature) was normalised at.

    The wavelength is a fitted-by-search number, not a constant of the method, so
    it is written beside the K tables instead of being implied by them: a
    normalised K can only be reproduced with the crossing that produced it.

    Parameters
    ----------
    ef : EnspireFile
        The parsed EnSpire export.
    note_fp : Path
        The note describing wells, titrant and samples.
    out_dir : Path
        Where ``<stem>_isosbestic.csv`` goes.
    isosbestic : float | str
        A wavelength [nm] or ``"auto"``.

    Returns
    -------
    pd.DataFrame
        One row per (sample, temperature): the wavelength used (``nan`` when the
        group has no crossing, which is the chloride case) and the wells read.
    """
    note = Note(note_fp).note
    samples = note[note["Name"] != "buffer"]
    buffers = note[note["Name"] == "buffer"]
    if isinstance(isosbestic, str) and isosbestic == "auto-scale":
        shared = isosbestic_by_group(ef, note)
        shared_rows = [
            {
                "sample": name,
                "temp": temp,
                "isosbestic_nm": shared.get((str(name), float(str(temp))), np.nan),
                "n_wells": int((samples["Name"] == name).sum()),
            }
            for (name, temp) in samples[["Name", "Temp"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ]
        table = pd.DataFrame(
            shared_rows, columns=["sample", "temp", "isosbestic_nm", "n_wells"]
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = note_fp.name.removesuffix("_note.csv").removesuffix("_joined_note")
        table.to_csv(out_dir / f"{stem}_isosbestic.csv", index=False)
        return table
    rows: list[dict[str, object]] = []
    for (name, temp), group in samples.groupby(
        ["Name", "Temp"], sort=True, dropna=False
    ):
        one = group.drop_duplicates("Well")
        is_ph = bool(one["pH"].nunique() > one["Cl"].nunique())
        wells = one["Well"].tolist()
        x = np.asarray(one["pH" if is_ph else "Cl"], dtype=np.float64)
        wanted_temp = float(str(temp))
        bufw = buffers[np.isclose(buffers["Temp"].astype(float), wanted_temp)][
            "Well"
        ].tolist()
        spectra, used, used_x, _peaks = sample_spectra(
            ef,
            wells,
            x,
            str(temp),
            is_ph=is_ph,
            buffer_wells=bufw,
            bands=("exc_anionic",),
        )
        # sample_spectra drops blank plate rows, so the crossing is located on
        # the axis it returns rather than on the note's
        lam_iso = (
            plate_isosbestic(
                spectra, used_x, x_min=bands_module.PH_ACID_END if is_ph else None
            )
            if isinstance(isosbestic, str)
            else isosbestic
        )
        rows.append({
            "sample": name,
            "temp": temp,
            "isosbestic_nm": lam_iso if lam_iso is not None else np.nan,
            "n_wells": len(used),
        })
    table = pd.DataFrame(rows, columns=["sample", "temp", "isosbestic_nm", "n_wells"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = note_fp.name.removesuffix("_note.csv").removesuffix("_joined_note")
    table.to_csv(out_dir / f"{stem}_isosbestic.csv", index=False)
    return table


@ppr.command(name="enspire-batch")
@click.pass_context
@click.argument("root", type=cPath(exists=True, file_okay=False, path_type=str))
@click.option("--buffer/--no-buffer", default=True, show_default=True, help="Subtract the buffer well of the same plate column.")  # fmt: skip
@click.option("--screen", default=bands_module.DEFAULT_SCREEN, show_default=True, help="Point screen as method:threshold:min_keep; 'none' fits every point.")  # fmt: skip
@click.option("--min-rho", default=0.8, show_default=True, help="Smallest |Spearman rho| for a band to carry K.")  # fmt: skip
def enspire_batch(
    ctx: Context, root: str, buffer: bool, screen: str, min_rho: float
) -> None:
    """Fit every folder under ROOT holding an EnSpire csv beside its *_note.csv.

    One row per fit lands in ``enspire_batch_K.csv`` and one per folder that
    could not be paired or fitted in ``enspire_batch_failures.csv``: a folder
    that fails is part of the result, not a silent gap.
    """
    out = Path(ctx.obj.get("OUT", __enspire_out_dir__))
    out.mkdir(parents=True, exist_ok=True)
    root_dir = Path(root)
    collected: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for note_fp in sorted(root_dir.rglob("*_note.csv")):
        data = _data_for(note_fp)
        if data is None:
            failures.append({"note": str(note_fp), "why": "no unambiguous data csv"})
            continue
        tag = note_fp.parent.relative_to(root_dir).as_posix().replace("/", "_")
        try:
            band_run = run_bands(
                EnspireFile(data),
                note_fp,
                out / tag,
                buffer=buffer,
                screen=None if screen.lower() == "none" else screen,
                min_rho=min_rho,
                wanted=bands_module.DIRECT_BANDS,
            )
            _echo_band_summaries(band_run)
            table = band_run.table
        except Exception as exc:  # ruff: ignore[blind-except] - a bad folder must not stop the walk
            failures.append({
                "note": str(note_fp),
                "why": f"{type(exc).__name__}: {exc}",
            })
            continue
        table.insert(0, "folder", note_fp.parent.relative_to(root_dir).as_posix())
        table.insert(1, "file", data.name)
        collected.append(table)
    if collected:
        every = pd.concat(collected, ignore_index=True)
        every.to_csv(out / "enspire_batch_K.csv", index=False)
        print(f"{len(collected)} files fitted -> {out / 'enspire_batch_K.csv'}")
    if failures:
        pd.DataFrame(failures).to_csv(out / "enspire_batch_failures.csv", index=False)
        print(f"{len(failures)} not fitted -> {out / 'enspire_batch_failures.csv'}")


def _data_for(note_fp: Path) -> Path | None:
    """Return the EnSpire csv a note belongs to, or None when ambiguous."""
    exact = note_fp.with_name(f"{note_fp.name.removesuffix('_note.csv')}.csv")
    if exact.exists():
        return exact
    others = [
        p for p in note_fp.parent.glob("*.csv") if not p.name.endswith("_note.csv")
    ]
    return others[0] if len(others) == 1 else None


#############################
#  fit_titration_global     #
#############################
@click.group()
@click.pass_context
@click.version_option(message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--out", "-o", type=cPath(), help="Output folder.")
@click.option(
    "--is-ph/--no-is-ph", default=True, show_default=True, help="Concentrations are pH."
)
def fit_titration(
    ctx: Context, verbose: int, out: str, is_ph: bool
) -> None:  # pragma: no cover
    """Fit Titration group command."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose
    ctx.obj["OUT"] = out
    ctx.obj["IS_PH"] = is_ph


@fit_titration.command()
@click.pass_context
@click.argument("csv_f", type=click.Path(exists=True))
@click.argument("note_f", type=click.Path(exists=True))
@click.option(
    "-b", "--band", nargs=2, type=int, help="Integration interval from <1> to <2>"
)
def spec(ctx: Context, csv_f: str, note_f: str, band: tuple[int, int] | None) -> None:
    """Update old svd or band fit of titration spectra."""
    verbose = ctx.obj.get("VERBOSE", 0)
    is_ph = ctx.obj.get("IS_PH", True)
    out = Path(ctx.obj.get("OUT", "."))

    note_df = pd.read_csv(note_f, sep="\t")
    csv_df = pd.read_csv(csv_f)
    # Ignore buffer wells! SVD will use differences between spectra.
    note_df = note_df[note_df["mutant"] != "buffer"]

    class Notes(NamedTuple):
        wells: list[str]
        conc: list[float]

    titration_type = "pH" if is_ph else "Cl"
    note = Notes(list(note_df["well"]), list(note_df[titration_type]))
    spectra = csv_df[note.wells]
    spectra.index = csv_df["lambda"]
    spectra.columns = np.array(note.conc)
    if verbose:
        print(csv_df)
        click.echo(note_f)
        print(note)
        print("DataFrame\n", spectra)
    is_ph = titration_type == "pH"
    fit_result = fitting.core.analyze_spectra(spectra, is_ph=is_ph, band=band)
    # output
    out.mkdir(parents=True, exist_ok=True)
    pdf_file = out / f"{Path(csv_f).stem}_{band}_{Path(note_f).stem}.pdf"
    if fit_result.figure is not None:
        fit_result.figure.savefig(pdf_file)
    click.echo(result_report(fit_result, pdf_file, str(band)))


@fit_titration.command()
@click.pass_context
@click.argument("file", type=click.Path(exists=True))
@click.option("-b", "--boot", type=int, help="Number of booting iterations.")
@click.option(
    "--weight/--no-weight", default=True, show_default=True, help="Use residue weights."
)
def glob(ctx: Context, file: str, boot: int, weight: bool) -> None:
    """Update old glob fit of multiple datasets."""
    verbose = ctx.obj.get("VERBOSE", 0)
    is_ph = ctx.obj.get("IS_PH", True)
    file_df = pd.read_csv(file)
    fp = Path(file)
    min_correl_to_print = 0.65
    burn = 75
    if verbose:
        click.echo(file_df)
    x = file_df["x"].to_numpy().astype(float)
    ds_data = {
        lbl: DataArray(x, file_df[lbl].to_numpy().astype(float))
        for lbl in file_df.columns[1:]
    }
    ds = Dataset(ds_data, is_ph=is_ph)
    if weight:
        fitting.core.weight_multi_ds_titration(ds)
    f_res = fitting.core.fit_binding_glob(ds)
    params = f_res.result.params if f_res.result else lmfit.Parameters()
    # Figure
    figure, ax = plt.subplots()
    fitting.plotting.plot_fit(
        ax, ds, params, nboot=30, pp=fitting.plotting.PlotParameters(is_ph)
    )
    lmfit.printfuncs.report_fit(f_res.result, min_correl=min_correl_to_print)
    figure.savefig(Path(file).with_suffix(".png"))
    if boot and f_res.mini:
        # Emcee
        samples = f_res.mini.emcee(burn=burn, steps=boot).flatchain
        fig = fitting.plotting.plot_emcee(samples)
        fig.savefig(fp.with_suffix(".png").with_stem(fp.stem + "-emcee"))
        hdi = samples.quantile([0.025, 0.975])["K"].to_list()
        print(f"Quantiles for K: {[f'{q:.3g}' for q in hdi]}")
        hdi = samples.quantile([0.03, 0.97])["K"].to_list()
        print(f"HDI (94%): {[f'{q:.3g}' for q in hdi]}")
        # R := S0 / S1
        # function := S1 * (R + (1 - R) * 10 ** (K - x) / (1 + 10 ** (K - x)))
        if is_ph:  # ratio between protonated un-protonated states
            ratios = {lbl: samples[f"S0_{lbl}"] / samples[f"S1_{lbl}"] for lbl in ds}
            # Combine ratio and K samples into a DataFrame for corner plot
            samples_ratios = pd.DataFrame({**ratios, "K": samples["K"]})
            fig_ratio = fitting.plotting.plot_emcee(samples_ratios)
            fig_ratio.savefig(fp.with_suffix(".png").with_stem(fp.stem + "-emc-ratios"))
            for lbl in ds:
                hdi = samples_ratios.quantile([0.025, 0.5, 0.975])[lbl].to_list()
                formatted_hdi = [f"{q:.3g}" for q in hdi]
                print(f"HDI (94%) for plateau ratio in dataset {lbl}: {formatted_hdi}")


########################################
# note_to_csv                          #
########################################
@click.command()
@click.argument("note", type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output CSV file.")
@click.option("-l", "--labels", default="A B", help="Labels to be appended.")
@click.option("-t", "--temp", default="37.0", help="Temperature to be appended.")
def note2csv(note: str, output: str, labels: str, temp: str) -> None:
    """Convert a tab-separated data file into a CSV file."""
    headers = ["Well", "pH", "Cl", "Name", "Temp", "Labels"]

    input_path = Path(note)
    output_path = Path(output) if output else input_path.with_suffix(".csv")

    if not output_path.exists():
        output_path.write_text(",".join(headers) + "\n", encoding="utf-8")

    # read data from note file, append to output file
    with input_path.open("r", encoding="utf-8") as datafile:
        reader = csv.reader(datafile, delimiter="\t")  # assuming tab-separated values
        next(reader)  # skip the header row

        with output_path.open("a", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            for row in reader:
                new_row = [*row[:4], temp, labels]
                writer.writerow(new_row)
