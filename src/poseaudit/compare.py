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
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        return text

    def summary(self, degree: str = "°") -> str:
        """Each model's table of measures, then the differences."""
        from poseaudit.report import table

        out = []
        for name, results in self.results.items():
            out += [f"{name}:", table(results, degree), ""]
        out.append(differences_table(self.differences, self.results, degree))
        return "\n".join(out)

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
        }


def compare(
    predictions: Mapping[str, Dataset],
    truth: Dataset,
    measures: Measure | Sequence[Measure],
    big_error: float,
    cluster: Cluster = None,
    resamples: int = 2000,
    seed: int = 0,
    min_iou: float = 0.3,
    min_keypoint_similarity: float = 0.5,
    **audit_options,
) -> Comparison:
    """Audit each model in `predictions` ({name: dataset}) against `truth` on
    every measure, and compare every pair of models on the readings both made
    of the same labelled instance. Other keyword arguments go to `audit`."""
    if len(predictions) < 2:
        raise ValueError("compare needs at least two models")
    chosen = [measures] if isinstance(measures, Measure) else list(measures)
    if not chosen:
        raise ValueError("compare needs at least one measure")
    results: dict[str, list[AuditResult]] = {}
    for name, predicted in predictions.items():
        pairing = pair(truth, predicted, min_iou, min_keypoint_similarity)
        results[name] = [
            audit(
                pairing,
                m,
                big_error,
                cluster=cluster,
                resamples=resamples,
                seed=seed,
                **audit_options,
            )  # fmt: skip
            for m in chosen
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
    )


def differences_table(differences, results, degree: str = "°") -> str:
    """The differences as aligned text, a minus b on the shared readings."""
    from poseaudit.report import _ci, _f, label

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
    widths = [max(len(r[k]) for r in rows) for k in range(len(head))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in rows
    )
