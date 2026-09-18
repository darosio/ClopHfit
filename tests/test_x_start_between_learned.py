"""How far apart the wells' pH axes sit: a prior, not something fluorescence can tell.

``x_start_between_sigma`` pins that spread, and a well's pH offset shifts its K
exactly as K itself does. With every well free the two are one parameter, so
``learn_x_start_between`` collapses towards zero however displaced the wells
really are - and takes K's interval down with it. Only replicates known to share
a K identify it, which is what ``K_sigma_w`` measures. These tests pin both
halves so the flag is not mistaken for an improvement on free-K plates.
"""

from __future__ import annotations

import numpy as np

from clophfit.fitting import bayes
from clophfit.fitting.bayes_config import SamplerConfig
from clophfit.fitting.data_structures import DataArray, Dataset
from clophfit.fitting.models import binding_1site
from clophfit.prtecan import PlateScheme

PH = np.array([8.9, 8.3, 7.7, 7.1, 6.5, 5.9, 5.3])
SAMPLER = SamplerConfig(n_samples=400, n_tune=400, chains=2, random_seed=1234)


def _plate(offset_sd: float, *, seed: int = 0) -> dict[str, Dataset]:
    """Twelve wells whose own pH axis is displaced by *offset_sd* pH."""
    rng = np.random.default_rng(seed)
    out: dict[str, Dataset] = {}
    for i in range(12):
        shift = rng.normal(0.0, offset_sd)
        arrays = {}
        for lbl, (s0, s1) in (("1", (200.0, 1000.0)), ("2", (1000.0, 200.0))):
            # The well is read on the nominal axis but titrated on a shifted one.
            y = binding_1site(PH + shift, 7.0, s0, s1, is_ph=True) + rng.normal(
                0.0, 4.0, PH.size
            )
            arrays[lbl] = DataArray(PH, y, y_errc=np.full(PH.size, 4.0))
        out[f"A{i:02d}"] = Dataset(arrays, is_ph=True)
    return out


def _fit(dsd: dict[str, Dataset], **kw: object) -> bayes.MultiFitResult:
    """Multi-well fit on the per-well pH axis."""
    return bayes.fit_binding_pymc_multi(
        dsd,
        PlateScheme(),
        x_error_model="per_well",
        x_start_between_sigma=0.05,
        sampler=SAMPLER,
        **kw,  # type: ignore[arg-type]
    )


def test_free_k_cannot_identify_the_spread_however_displaced_the_wells() -> None:
    """With every well free, its pH offset and its K are the same parameter.

    Shifting a well's axis shifts its curve exactly as a change in K does, so
    the likelihood is flat along that ridge: the fit puts the displacement in K
    and leaves the axis term near zero, whatever the truth. The pinned
    ``x_start_between_sigma`` is therefore an informative prior - it passes a
    known pH dispersion into K's interval - and not a value the fluorescence can
    estimate.
    """
    wide = _fit(_plate(0.12, seed=1), learn_x_start_between=True)
    tight = _fit(_plate(0.0, seed=2), learn_x_start_between=True)
    assert wide.trace is not None
    assert tight.trace is not None
    sw = float(wide.trace.posterior["x_start_between"].mean())
    st = float(tight.trace.posterior["x_start_between"].mean())
    assert sw < 0.05  # nowhere near the 0.12 planted
    assert abs(sw - st) < 0.03


def test_replicates_that_share_a_k_do_identify_it() -> None:
    """Six wells known to be the same sample: their axes' spread is then estimable.

    A control group pins the K the replicates share, so a well's remaining
    displacement can only be its own axis - the same information ``K_sigma_w``
    reads between replicates.
    """
    dsd = _plate(0.12, seed=5)
    scheme = PlateScheme()
    scheme.names = {"ctrl": set(list(dsd)[:6])}
    res = bayes.fit_binding_pymc_multi(
        dsd,
        scheme,
        x_error_model="per_well",
        x_start_between_sigma=0.2,
        learn_x_start_between=True,
        sampler=SAMPLER,
    )
    assert res.trace is not None
    assert float(res.trace.posterior["x_start_between"].mean()) > 0.05


def test_pinning_it_leaves_no_such_variable() -> None:
    """The default is unchanged: a fixed scale, and x_start_well still per well."""
    res = _fit(_plate(0.12, seed=3))
    assert res.trace is not None
    assert "x_start_between" not in res.trace.posterior
    assert "x_start_well" in res.trace.posterior


def test_learning_it_narrows_k_because_the_spread_collapses() -> None:
    """The hazard: on free-K wells, learning the scale halves K's interval.

    The pinned scale is what passes pH dispersion into K's uncertainty; once the
    unidentified scale collapses towards zero, that contribution goes with it and
    the interval shrinks - the opposite of what a well-to-well spread should do.
    Learn it only where replicates pin the K (see the test above), and otherwise
    pin it at the dispersion measured there.
    """
    dsd = _plate(0.12, seed=4)
    sk = {}
    for name, kw in (("pinned", {}), ("learned", {"learn_x_start_between": True})):
        trace = _fit(dsd, **kw).trace
        assert trace is not None
        sk[name] = float(trace.posterior["K_free"].std(("chain", "draw")).mean())
    assert sk["learned"] < 0.75 * sk["pinned"]
