"""Command-line interface."""

from __future__ import annotations

import csv
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
from clophfit.prenspire import EnspireFile, Note
from clophfit.prtecan import McmcSpec, TecanConfig, Titration, calculate_conc
from clophfit.prtecan.export import export_data_fit

if TYPE_CHECKING:
    from clophfit.fitting.data_structures import FitResult


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
@click.option("--sch", type=cPath(exists=True), help="Path to plate scheme file (buffers and controls).")  # fmt: skip
@click.option("--add", type=cPath(exists=True), help="Path to additions file (initial volume + additions); enables dilution correction.")  # fmt: skip
@click.option("--all", "comb", is_flag=True, help="Export all bg/dil/nrm data combinations.")  # fmt: skip
@click.option("--lim", type=(float, float), help="x-axis range MIN MAX for K plots.")
@click.option("--title", "-t", type=str, default="", help="Title for plots.")
@click.option("--fit/--no-fit", default=True, show_default=True, help="Perform fitting.")  # fmt: skip
@click.option("--png/--no-png", default=True, show_default=True, help="Export PNG files.")  # fmt: skip
@click.option("--fit-method", default="huber", show_default=True, type=click.Choice(["lm", "huber", "irls", "odr"], case_sensitive=False), help="Global fit method: lm (standard LS), huber (robust Huber loss), irls (iterative reweighting), odr (orthogonal distance regression, x-aware).")  # fmt: skip
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
@click.option("--per-well-ye-mags/--no-per-well-ye-mags", "per_well_ye_mags", default=None, help="For --mcmc multi: scale y_err per well rather than per label. Unset lets the library resolve it from the noise family, which couples the two.")  # fmt: skip
@click.option("--ye-mag-parameterization", type=click.Choice(["centered", "hierarchical", "separable", "separable_step"], case_sensitive=False), default="centered", show_default=True, help="For --mcmc multi with per-well ye_mags: independent per label (centered), a shared well factor with per-label deviations (hierarchical), a per-label level plus one shared well factor (separable), or that plus a per-label pH axis on the noise (separable_step).")  # fmt: skip
@click.option("--plate-fit", type=click.Choice(["lm", "odr"], case_sensitive=False), default=None, help="Also fit the whole plate in one classical least-squares problem, with the noise scale profiled per label across the plate and each control group pooled onto one K. Writes plate_{method}_K.csv. Minutes rather than hours, and as accurate against known pKs as the sampler.")  # fmt: skip
@click.option("--plate-noise", type=click.Choice(["fixed", "calibrated"], case_sensitive=False), default="fixed", show_default=True, help="How --plate-fit weights the points K is fitted to, with or without --plate-screen-z. fixed uses y_err as built (bg_noise floor plus any --noise-gain/--noise-alpha). calibrated estimates gain and alpha per label from the fit's own residuals and refits under them; it describes the residuals better and fits K worse, so it is not the default.")  # fmt: skip
@click.option("--plate-screen-noise", type=click.Choice(["calibrated", "fixed"], case_sensitive=False), default="calibrated", show_default=True, help="The ruler --plate-screen-z judges points on. calibrated fits gain and alpha to the screening pass's own residuals so bright and dim points are judged alike; fixed judges on y_err as built. Separate from --plate-noise, which sets the weights K is then fitted with.")  # fmt: skip
@click.option("--plate-screen-z", type=float, default=None, help="For --plate-fit: drop points whose |z| exceeds this and refit. The ruler is --plate-screen-noise (calibrated by default, so a dim point and a bright one are judged on the same scale); the refit uses --plate-noise's weights. 3.0 is the value measured to help; 2.5 is harmful.")  # fmt: skip
@click.option("--plate-screen-frac", type=float, default=None, help="For --plate-fit: also drop 400 nm points whose |y-yhat|/yhat exceeds this, sparing any the 485 nm channel moves with. A z-score fails at both ends of a titration -- sigma tracks the signal while model error tracks the curve -- so a 5% miss at the dim end reads as 3.6 sigma while a 36% miss at the bright end reads as 2.7. 0.12 is where reviewer calls separate; unset leaves the z-screen alone.")  # fmt: skip
@click.option("--mcmc-robust/--no-mcmc-robust", "mcmc_robust", default=False, show_default=True, help="Use a robust likelihood for --mcmc instead of a Normal. Student-t nu=3 was the best-calibrated arm on this campaign's plates.")  # fmt: skip
@click.option("--mcmc-robust-likelihood", type=click.Choice(["student_t", "mixture"], case_sensitive=False), default="student_t", show_default=True, help="Which robust likelihood --mcmc-robust selects: a heavy-tailed student_t, or a Normal/outlier contamination mixture that models the outliers rather than down-weighting them.")  # fmt: skip
@click.option("--student-t-nu", default=3.0, show_default=True, type=float, help="Student-t degrees of freedom for --mcmc-robust. Lower is heavier-tailed; pass 0 to infer nu (support above 2).")  # fmt: skip
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
    sch: str | None,
    add: str | None,
    comb: bool,
    lim: tuple[float, float] | None,
    title: str,
    fit: bool,
    png: bool,
    fit_method: str,
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
    mcmc_x_error: str,
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

    if print_spec:
        _echo_spec({
            "bg": bg,
            "bg_adj": bg_adj,
            "bg_mth": bg_mth,
            "nrm": nrm,
            "dil": dil,
            "fit_method": fit_method,
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
            "ye_mag_parameterization": ye_mag_parameterization,
            "mcmc_robust": mcmc_robust,
            "student_t_nu": student_t_nu if mcmc_robust else None,
            "ctr_free_k": ctr_free_k,
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
            structured_noise=mcmc_noise == "structured",
            per_well_ye_mags=per_well_ye_mags,
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
def enspire(ctx: Context, csv_f: str, note_f: str | None, bands: tuple[Any]) -> None:
    """Save spectra as csv tables from EnSpire xls file."""
    verbose = ctx.obj.get("VERBOSE", 0)
    out = ctx.obj.get("OUT", __enspire_out_dir__)
    ef = EnspireFile(Path(csv_f), verbose=verbose)
    ef.export_measurements(Path(out))
    if note_f is not None:
        fit_enspire(ef, Path(note_f), Path(out), list(bands), verbose)


# TODO: Simplify this function
def fit_enspire(  # ruff: ignore[complex-structure, too-many-branches]
    ef: EnspireFile,
    note_fp: Path,
    out_dir: Path,
    bands: list[tuple[str, int, int]],
    verbose: int,
) -> None:
    """Fit prenspire titration (all labels, temp, mutant, titrations)."""
    note = Note(note_fp, verbose=verbose)
    note.build_titrations(ef)
    dbands = {label: (ini, fin) for label, ini, fin in bands} if bands else {}
    ds_data: dict[str, DataArray] = {}
    for name, d_name in note.titrations.items():
        for temp, d_temp in d_name.items():
            for tit, d_tit in d_temp.items():
                if tit.split("_")[0] == "pH":
                    is_ph = False
                elif tit.split("_")[0] == "Cl":
                    is_ph = True
                else:
                    msg = "Unknown titration type."
                    raise ValueError(msg)
                for label, data in d_tit.items():
                    band = dbands.get(label)
                    fit_result = fitting.core.analyze_spectra(
                        data, is_ph=is_ph, band=band
                    )
                    if fit_result.is_valid() and fit_result.mini:
                        userargs = fit_result.mini.userargs[0]["default"]
                        ds_data[label] = DataArray(userargs.x, userargs.y)
                        pdf_file = out_dir / f"{name}_{temp}_{label}_{tit}_{band}.pdf"
                        if fit_result.figure:
                            fit_result.figure.savefig(pdf_file)
                    _print_result(fit_result, pdf_file, str(band))
                # Global spectra analysis with more than 1 label.
                if (
                    len(d_tit.keys() - dbands.keys()) > 1  # svd > 1
                    or len(dbands.keys() & d_tit.keys()) > 1  # bands > 1
                ):
                    ds = Dataset(ds_data, is_ph=is_ph)
                    spectra_gres = fitting.core.analyze_spectra_glob(d_tit, ds, dbands)
                    if spectra_gres.svd and spectra_gres.svd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_SVD.pdf"
                        if spectra_gres.svd.figure:
                            spectra_gres.svd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.svd, pdf_file, "")
                    if spectra_gres.gsvd and spectra_gres.gsvd.is_valid():
                        pdf_file = out_dir / f"{name}_{temp}_g_{tit}_SVD.pdf"
                        if spectra_gres.gsvd.figure:
                            spectra_gres.gsvd.figure.savefig(pdf_file)
                        _print_result(spectra_gres.gsvd, pdf_file, "")
                    if spectra_gres.bands and spectra_gres.bands.is_valid():
                        keys = dbands.keys() & d_tit.keys()
                        lname = [f"{k}({dbands[k][0]},{dbands[k][1]})" for k in keys]
                        bands_str = "".join(lname)
                        pdf_file = out_dir / f"{name}_{temp}_all_{tit}_{bands_str}.pdf"
                        if spectra_gres.bands.figure:
                            spectra_gres.bands.figure.savefig(pdf_file)
                        _print_result(spectra_gres.bands, pdf_file, bands_str)


def _print_result(fit_result: FitResult, pdf_file: Path, band_str: str) -> None:
    print(str(pdf_file))
    print(f"Best fit using '{band_str}' band:\n")
    ci = lmfit.conf_interval(fit_result.mini, fit_result.result)
    print(lmfit.ci_report(ci, ndigits=2, with_offset=False))
    print(f"\n Plot saved in '{pdf_file}'.\n")


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
    _print_result(fit_result, pdf_file, str(band))


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
