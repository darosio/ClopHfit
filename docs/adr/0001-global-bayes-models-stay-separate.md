# 0001. The two whole-spectrum Bayesian models stay separate

Status: accepted (2026-11-08)

## Context

`fit_spectra_global_bayes` (free species spectra) and
`fit_spectra_global_bayes_marginal` (species integrated out) live in
`clophfit.fitting.spectral`. They share the mixture model and the reporting,
but their bodies differ: they share about a third of their lines, and the
species treatment, priors and sampler graph differ. A single function that
takes a "species prior" argument would rewrite two different PyMC models.

The EnSpire runner (`clophfit.prenspire.runs.global_bayes`) already runs both
through one pipeline by passing the fitter in, so the duplication is in the
modelling code only.

## Decision

Keep the two models as separate functions. Do not merge them into one
parameterised model until a benchmark comparison (see `AGENTS.md`) has chosen
which formulation to keep.

## Consequences

- No change to `fitting/spectral.py`.
- The shared runner stays the single place where both models are invoked from
  the CLI.
- Revisit when the benchmark comparison is done; if one model is retired, the
  other stays as is and the runner loses one fitter argument.
