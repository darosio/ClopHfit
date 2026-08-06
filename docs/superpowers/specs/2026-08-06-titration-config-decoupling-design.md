# Decoupling TitrationConfig from Bayesian vocabulary

- **Date:** 2026-08-06
- **Status:** approved, pending implementation plan
- **Scope:** `clophfit.prtecan.titration` (`TitrationConfig`),
  `clophfit.prtecan.export`, `clophfit.__main__`

## Problem

`TitrationConfig` (`titration.py:121`) is documented as "Parameters defining the
fitting data". Six of its seventeen fields do not define fitting data: they
define how a sampler runs. They were added to carry CLI options to code three
call-levels down, and the coupling is now visible in three places at once.

**The fields name their own CLI flags.** `mcmc_noise` (`titration.py:154`) is
documented as "Observation-noise family for `--mcmc single-refit`". `noise_alpha`
(`titration.py:135`) says "Values typically from MCMC multi-noise
shared_noise_params.csv". A library dataclass should not be describing a command
line.

**The library dispatches on the mode strings.** The CLI writes them onto the
config, and `export.py` branches on them:

| Site              | Code                                                       |
| ----------------- | ---------------------------------------------------------- |
| `__main__.py:277` | `tit.params.mcmc = mcmc`                                   |
| `__main__.py:283` | `tit.params.mcmc_noise = mcmc_noise`                       |
| `export.py:269`   | `if titration.params.mcmc == "single":`                    |
| `export.py:282`   | `if titration.params.mcmc != "single-refit":`              |
| `export.py:289`   | `structured = titration.params.mcmc_noise == "structured"` |
| `export.py:155`   | `supplied_mode = titration.params.noise_mode`              |

A string typed at a terminal is parsed by string comparison inside the export
layer. Adding a sampling variant means adding a `_FlexChoice` entry
(`__main__.py:157`) and another equality test in `export.py`.

**Half the vocabulary resolves to nothing.** `params.mcmc` is read in exactly
two places, both shown above, and both compare against `"single"` or
`"single-refit"`. Every other value makes `fit_single_mcmc` return `None`, so
`--mcmc multi`, `--mcmc multi-noise` and `--mcmc multi-noise-xrw` are
behaviourally identical to `--mcmc None` while still being advertised in
`--help`.

Commit `01735f12` (2026-05-19, "refactor: extract fitting export logic out of
Titration") is the cause. It removed 724 lines from `titration.py` against 202
added to `export.py`, deleting the whole multi-MCMC path — the
`if self.params.mcmc == "multi-noise"` dispatch, the per-mode `stage_map`, and
the cached properties `result_mcmc`, `result_multi_trace`,
`result_multi_trace2`, `result_multi_noise`, `result_multi_noise_xrw`,
`result_multi_mcmc`, `result_multi_noise_mcmc`, `result_multi_noise_xrw_mcmc`.
`export.py` gained only `collect_multi_residuals` in exchange. The path was
dropped, not moved.

Two orphans were left behind. The three CLI choices above, and the tuple in
`_reset_data_and_results` (`titration.py:919`), which still does
`self.__dict__.pop(attr, None)` over **twelve** names — the eight cached
properties above plus `results`, `result_global`, `result_odr` and
`fit_pipeline`. Not one of them exists: `Titration` has exactly one
`cached_property`, `fit_keys` (`titration.py:788`), and it is not in the tuple.
The `.results` hits elsewhere belong to `TitrationResults`, a different class;
`result_global` survives only in a stale docstring at `plotting.py:1340`
("Normally this is `tit.result_global.results`"). So the method's twelve pops
are all no-ops and its real work is the single line `self._data = {}`.
(`fit_pipeline` matches the 2026-07-18 plate-refit spec's note that
`clophfit.fitting.pipeline` was dissolved.)

The capability itself is intact and better off: `fit_binding_pymc_multi` takes
the composable config objects and is called by `ctr_validation.py`,
`plotting.py`, the notebook, and `arslanbaeva` (via its
`fit_function = "clophfit.fitting.bayes:fit_binding_pymc_multi"` config string).
It has zero references in `prtecan/` or `__main__.py`: what was lost is the
terminal route, not the science.

**The destination already exists.** `bayes_config.py` defines `SamplerConfig`
(`:66`), `RobustConfig` (`:134`), `InitConfig` (`:166`) and `NoiseConfig`
(`:199`). `prtecan_devel.ipynb` imports all four and passes them straight to
`fit_binding_pymc_multi`. The composable form is what the library and the
scratchpad already use; only the CLI path still speaks in mode strings.

## What stays and what leaves

The boundary is whether a field changes the **prepared dataset**.

| Stays (shapes data or the classical fit)      | Leaves (sampler vocabulary) |
| --------------------------------------------- | --------------------------- |
| `bg`, `bg_adj`, `dil`, `nrm`, `bg_mth`        | `mcmc` (`:131`)             |
| `fit_method`, `outlier`                       | `nuts_sampler` (`:132`)     |
| `mask_outliers`, `outlier_threshold`          | `n_mcmc_samples` (`:133`)   |
| `noise_alpha` (`:135`), `noise_gain` (`:145`) | `ctr_free_k` (`:134`)       |
|                                               | `mcmc_noise` (`:154`)       |
|                                               | `noise_mode` (`:162`)       |

`noise_alpha` and `noise_gain` stay despite their MCMC-flavoured docstrings.
They are the noise model itself: `_apply_error_model` (`titration.py:1181`)
reads them at `:1189` and `:1194` to build a `PlateNoiseModel`, which sets the
Dataset's `y_err`. That happens upstream of every fit, classical or Bayesian.
Their *provenance* is often a Bayesian fit; their *application* is universal.

`mcmc_noise` and `noise_mode` leave because they describe how a sampler treats a
supplied value — centre a prior on it, or pin it — which is a property of the
sampling call, not of the plate.

## Design

**Destinations.** `nuts_sampler` and `n_mcmc_samples` already exist as fields on
`SamplerConfig`; the `TitrationConfig` copies are deleted, not moved.
`mcmc_noise` and `noise_mode` become arguments of `NoiseConfig` construction.
`ctr_free_k` is already a keyword argument of `fit_binding_pymc_multi`, so the
field is redundant and is deleted.

**The mode string stops being data.** `mcmc` becomes the caller's choice of
which function to call; `export.py` receives that decision already made. Only
the two live values survive the move, so `McmcSpec` models `single` and
`single-refit` and nothing else. The three dead choices are removed from the
`_FlexChoice` list at `__main__.py:157`; the twelve dead names go with the whole
tuple, reducing `_reset_data_and_results` to `self._data = {}`; and the stale
`tit.result_global` reference in `plotting.py:1340` is corrected.

This is a deletion, not a feature removal: nothing that works today stops
working. CLI access to multi-well fitting returns in sub-project 2 as
`ppr tecan mcmc --model multi`, built on `fit_binding_pymc_multi` and the config
objects rather than on the cached properties deleted in May.

**The seam.** `export_data_fit(titration, tecan_config)` currently performs
classical and Bayesian work, selecting between them by reading
`titration.params`. It gains an explicit optional parameter carrying the
sampling decision and its config objects:

```python
def export_data_fit(
    titration: Titration,
    tecan_config: TecanConfig,
    mcmc: McmcSpec | None = None,   # None -> classical only
) -> None: ...
```

`McmcSpec` is a small frozen dataclass holding the model choice and the
`SamplerConfig` / `NoiseConfig` / `RobustConfig` instances. After this change
`export.py` reads no `params` field that concerns sampling.

**The CLI keeps its shape.** `__main__.py:169` keeps every flag it has today,
including its four `noqa` suppressions; the only surface change is that `--mcmc`
stops offering three values that resolve to nothing. Its body stops assigning to
`tit.params` for the six departing fields and builds an `McmcSpec` instead.
Splitting the command into subcommands is sub-project 2; doing it here would
mean building new subcommands on vocabulary this spec deletes.

## Consequences

**Spurious recomputation goes away.** `TitrationConfig.__setattr__`
(`titration.py:187`) fires the callback registered at `titration.py:784`,
`_reset_data_results_and_bg`, which discards `_data`, `bg` and `bg_err`. Today,
setting `nuts_sampler` — a choice of NUTS *backend* — throws away the plate's
prepared data and background, forcing full recomputation on next access.
Deleting the six fields deletes six ways to trigger that.

Nothing needed is lost, because no fit results are cached on `Titration` at all
since `01735f12`: `fit_plate` returns a `TitrationResults` to its caller. This
consequence therefore argues *for* the change rather than qualifying it, but it
is still a behaviour change and is asserted in a test.

**Roundtrip loss.** Nothing serialises `TitrationConfig` to disk in a form that
would break, and `arslanbaeva` sets none of the six fields (it constructs its
own kwargs in `_common.build_fit_kwargs`). Hard removal, no deprecation shim.

## Testing

- The existing suite is the primary net: `tests/test_cli.py` exercises four
  `tecan` invocations, and the fitting tests cover `export_data_fit`.
- One new test pins the seam: `export_data_fit` with `mcmc=None` performs no
  sampling, and with an `McmcSpec` it samples without reading `params`.
- One new test pins the reset behaviour: setting a surviving field (`nrm`)
  clears `_data`/`bg`/`bg_err`, and the removed sampler fields no longer exist
  to trigger it. Asserting the *effect* rather than a list of attribute names is
  what keeps this from drifting into another twelve-name fiction.
- `--mcmc multi` must now be rejected by click rather than silently doing
  nothing; assert the non-zero exit and the error naming the valid choices.

## Out of scope

Deliberately excluded, each its own spec:

2. **Split `ppr tecan` into subcommands** (`fit`, `mcmc`, `qc`) over one shared
   plate-loading option group, removing the four `noqa` suppressions. This is
   where CLI access to multi-well fitting is restored, as
   `ppr tecan mcmc --model multi` over `fit_binding_pymc_multi` — replacing the
   route deleted in `01735f12`. Also the right home for fixing
   `--noise-alpha`/`--noise-gain`, which are bare floats whose help text
   instructs the user to transcribe numbers out of `shared_noise_params.csv` — a
   manual bridge around `noise_calibration.calibrate_noise_robust`
   (`noise_calibration.py:61`).
1. **`ppr tecan grid`** — config expansion, `--offset`/`--limit` sharding, and a
   `--resume` that validates output integrity instead of trusting file
   existence.
1. **Migrate `arslanbaeva`** — promote `bayes-compare`, `multi-refit` and
   `residual-tables` into subcommands, then delete `loaders.py` and
   `_common.build_fit_kwargs`, which re-implement `Titration` loading and
   `_apply_error_model`.
1. **Cross-check noise-model provenance** — compare gain/alpha learned inside a
   PyMC mixture fit against `calibrate_noise_robust` applied to that same fit's
   residuals. Both sides exist; nothing connects them, and the agreement has
   never been measured. New capability, not cleanup.

## Note on stale references

`noise_calibration.py` is recent. The 2026-07-18 plate-refit spec cites
`calibrate_noise_robust` at `pipeline.py:64` and `fit_noise_model_nnls` at
`utils.py:446`; both now live in `noise_calibration.py` (`:61`, `:224`).
`prtecan_devel.ipynb`'s final cell calls `utils.fit_gain_from_residuals`, which
does not exist — `utils` has only `reweight_from_residuals`. When using the
notebook as a reference for the target API, expect parts of it to predate the
refactors.
