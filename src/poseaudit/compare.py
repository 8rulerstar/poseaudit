"""Models compared on the readings they share.

Each model is audited as usual. For each measure and pair of models, the
readings both made of the same labelled instance are lined up, and the
differences in mean |error| and in the large-error rate get a paired bootstrap
interval that resamples whole images (or named clusters), so the same images
are drawn for both models every time.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np

from poseaudit.audit import AuditResult, Cluster, audit
from poseaudit.confidence import clustered_bootstrap
from poseaudit.measures import Measure
from poseaudit.pairing import pair
from poseaudit.types import Dataset

NAN2 = (float("nan"), float("nan"))
UNITS = {"deg": "degrees", "px": "px"}


@dataclass(frozen=True)
class Difference:
    """Model `a` minus model `b` on the readings both made: below 0, `a` is
    the closer to the truth."""

    measure: str
    a: str
    b: str
    n_shared: int  # labelled instances both models read
    n_a: int  # readings of `a` in all
    n_b: int
    clusters: int  # images (or named clusters) among the shared readings
    mean_abs_error_a: float
    mean_abs_error_b: float
    mean_abs_error_diff: float
    mean_abs_error_diff_ci: tuple[float, float]
    big_error_rate_a: float
    big_error_rate_b: float
    big_error_rate_diff: float
    big_error_rate_diff_ci: tuple[float, float]
    # shared readings the two models read differently: the intervals rest on these
    n_differing: int | None = None


# fewer shared readings read differently than this make the paired intervals
# rest on a handful of values
FEW_DIFFERING = 20


@dataclass
class Comparison:
    """What `compare` returns: each model's results, one per measure, and the
    differences between every pair of models."""

    results: dict[str, list[AuditResult]]
    differences: list[Difference] = field(default_factory=list)

    def table(self) -> list[dict]:
        """One flat row per measure and pair of models, intervals split into
        `*_ci_low` and `*_ci_high`."""
        rows = []
        for d in self.differences:
            row = asdict(d)
            for key in ("mean_abs_error_diff_ci", "big_error_rate_diff_ci"):
                low, high = row.pop(key)
                row[key + "_low"], row[key + "_high"] = low, high
            rows.append(row)
        return rows

    def to_csv(self, path: str | None = None) -> str:
        rows = self.table()
        buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(
                buffer, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        text = buffer.getvalue()
        if path:
            from poseaudit._files import write_text

            write_text(path, text, newline="")
        return text

    def summary(self, degree: str = "°", width: int | None = None) -> str:
        """Each model's table of measures, then the differences. With
        `width`, tables wider than that are cut into blocks of columns."""
        from poseaudit.report import table

        out = []
        for name, results in self.results.items():
            out += [f"{name}:", table(results, degree, width), ""]
        out.append(differences_table(self.differences, self.results, degree, width))
        return "\n".join(out)

    def __repr__(self) -> str:
        names = ", ".join(self.results)
        measures = len(next(iter(self.results.values()), []))
        pairs = len(self.differences)
        return (
            f"<Comparison of {names}: {measures} "
            f"{'measure' if measures == 1 else 'measures'}, "
            f"{pairs} {'difference' if pairs == 1 else 'differences'}>"
        )

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        """IPython and Jupyter show the summary, not every reading."""
        printer.text(repr(self) if cycle else self.summary())

    def to_dict(self) -> dict:
        from poseaudit._version import __version__

        first = next(iter(self.results.values()))
        return {
            "poseaudit": __version__,
            "schema": 1,
            "settings": first[0].settings if first else {},
            "models": {
                name: [r.to_dict() for r in results]
                for name, results in self.results.items()
            },
            "differences": self.table(),
            "warnings": difference_warnings(self.differences),
        }


def difference_warnings(differences) -> list[str]:
    """What makes a paired interval missing or too narrow to trust."""
    notes = []
    for d in differences:
        who = f"{d.measure}, {d.a} - {d.b}"
        if d.n_shared == 0:
            notes.append(f"{who}: no labelled instance was read by both models.")
        elif d.clusters < 2:
            notes.append(
                f"{who}: the shared readings come from one image or cluster, so "
                "there are no paired intervals."
            )
        elif d.clusters < 20:
            notes.append(
                f"{who}: the shared readings come from only {d.clusters} images "
                "or clusters: the paired intervals are too narrow with this few."
            )
        if d.n_shared and d.n_differing == 0:
            notes.append(
                f"{who}: the models read all {d.n_shared} shared readings the "
                "same, so every difference is 0 and its interval a point."
            )
        elif d.n_shared and d.n_differing is not None and d.n_differing < FEW_DIFFERING:
            rest = "that one" if d.n_differing == 1 else "those"
            notes.append(
                f"{who}: the models read only {d.n_differing} of the {d.n_shared} "
                f"shared readings differently; the paired intervals rest on {rest} "
                "and can be far too narrow."
            )
    return notes


def big_error_for(measure: Measure, big_error: float | Mapping[str, float]) -> float:
    """`big_error` itself, or from a mapping its value for the measure's kind
    (angle, tilt, length, ratio), else for its unit (deg, px)."""
    if not isinstance(big_error, Mapping):
        return big_error
    kind = type(measure).__name__.lower()
    for key in (kind, measure.unit):
        if key and key in big_error:
            return big_error[key]
    raise ValueError(
        f"big_error gives no value for {measure.name} {measure.points}: add "
        f"{kind!r} to it"
    )


def compare(
    predictions: Mapping[str, Dataset],
    truth: Dataset,
    measures: Measure | Sequence[Measure],
    big_error: float | Mapping[str, float],
    cluster: Cluster = None,
    resamples: int = 2000,
    seed: int = 0,
    min_iou: float = 0.3,
    min_keypoint_similarity: float = 0.5,
    **audit_options,
) -> Comparison:
    """Audit each model in `predictions` ({name: dataset}) against `truth` on
    every measure, and compare every pair of models on the readings both made
    of the same labelled instance. `big_error` is one value, or with measures
    in different units a mapping by kind or unit such as
    `{"angle": 15, "length": 10, "ratio": 0.1}`. Other keyword arguments go
    to `audit`."""
    if len(predictions) < 2:
        raise ValueError("compare needs at least two models")
    chosen = [measures] if isinstance(measures, Measure) else list(measures)
    if not chosen:
        raise ValueError("compare needs at least one measure")
    units = list(dict.fromkeys(UNITS.get(m.unit, m.unit or "ratios") for m in chosen))
    if not isinstance(big_error, Mapping) and len(units) > 1:
        raise ValueError(
            f"big_error {big_error:g} would count the same number as large in "
            f"{', '.join(units[:-1])} and {units[-1]} alike; pass one per kind of "
            "measure, such as "
            "{'angle': 15, 'length': 10, 'ratio': 0.1}"
        )
    bigs = [big_error_for(m, big_error) for m in chosen]
    results: dict[str, list[AuditResult]] = {}
    for name, predicted in predictions.items():
        pairing = pair(truth, predicted, min_iou, min_keypoint_similarity)
        results[name] = [
            audit(
                pairing,
                m,
                big,
                cluster=cluster,
                resamples=resamples,
                seed=seed,
                **audit_options,
            )  # fmt: skip
            for m, big in zip(chosen, bigs, strict=True)
        ]
    differences = [
        _difference(results[a][k], results[b][k], a, b, resamples, seed)
        for k in range(len(chosen))
        for a, b in combinations(results, 2)
    ]
    return Comparison(results, differences)


def _difference(ra, rb, a, b, resamples, seed) -> Difference:
    from poseaudit.report import label

    def by_instance(result):
        return {(r.image, r.truth_index): r for r in result.readings}

    left, right = by_instance(ra), by_instance(rb)
    shared = sorted(set(left) & set(right))
    ea = np.array([abs(left[k].error) for k in shared])
    eb = np.array([abs(right[k].error) for k in shared])
    groups = [left[k].cluster for k in shared]
    differing = sum(left[k].error != right[k].error for k in shared)
    clusters = len(set(groups))
    big = ra.big_error
    if shared:
        mae_a, mae_b = float(ea.mean()), float(eb.mean())
        rate_a, rate_b = float((ea >= big).mean()), float((eb >= big).mean())
    else:
        mae_a = mae_b = rate_a = rate_b = float("nan")
    mae_ci, rate_ci = NAN2, NAN2
    if clusters >= 2:
        diff = ea - eb
        flags = (ea >= big).astype(float) - (eb >= big).astype(float)
        low, high = clustered_bootstrap(
            groups,
            lambda idx: np.array([diff[idx].mean(), flags[idx].mean()]),
            resamples=resamples,
            seed=seed,
        )
        mae_ci = (float(low[0]), float(high[0]))
        rate_ci = (float(low[1]), float(high[1]))
    return Difference(
        measure=label(ra),
        a=a,
        b=b,
        n_shared=len(shared),
        n_a=ra.n,
        n_b=rb.n,
        clusters=clusters,
        mean_abs_error_a=mae_a,
        mean_abs_error_b=mae_b,
        mean_abs_error_diff=mae_a - mae_b,
        mean_abs_error_diff_ci=mae_ci,
        big_error_rate_a=rate_a,
        big_error_rate_b=rate_b,
        big_error_rate_diff=rate_a - rate_b,
        big_error_rate_diff_ci=rate_ci,
        n_differing=int(differing),
    )


def differences_table(
    differences, results, degree: str = "°", width: int | None = None
) -> str:
    """The differences as aligned text, a minus b on the shared readings."""
    from poseaudit.report import _ci, _f, label, layout

    units = {label(r): r.measure.unit for r in next(iter(results.values()))}
    unit_of = {"deg": degree, "px": " px"}
    head = ["measure", "a - b", "shared", "n a", "n b", "mean |error| a - b",
            "large-error rate a - b"]  # fmt: skip
    rows = [head]
    for d in differences:
        u = unit_of.get(units[d.measure], "")
        rows.append(
            [
                d.measure,
                f"{d.a} - {d.b}",
                str(d.n_shared),
                str(d.n_a),
                str(d.n_b),
                f"{_f(d.mean_abs_error_diff, '+.2f')}{u} "
                f"{_ci(d.mean_abs_error_diff_ci, '+.2f')}",
                f"{_f(d.big_error_rate_diff * 100, '+.1f')} pt "
                f"{_ci(tuple(100 * x for x in d.big_error_rate_diff_ci), '+.1f')}",
            ]
        )
    lines = layout(rows, frozenset({2, 3, 4}), keep=2, width=width)
    lines += [f"  ! {note}" for note in difference_warnings(differences)]
    return "\n".join(lines)
