"""How often does `budget`'s check pass a right fit, and how far apart are
the sizes it gives, at 100, 300 and 1,000 readings a group?

Synthetic bent arms (segments 10 to 200 px, 60 to 140 degrees) are read with
errors the fit can describe exactly: 2 px plus 9% of the limb, per axis, at
every point. Each case draws 40 data sets; the table gives the share whose
check passed, and the 5th, 50th and 95th percentiles of the sizes given.
A large error is 15 degrees; the target is 20%.

    uv run python validation/budget_power.py > validation/budget_power.out.md

Seeded, so the table is the same on every machine with the same numpy.
"""

import warnings

import numpy as np

from poseaudit import angle
from poseaudit.budget import budget
from poseaudit.pairing import Pair, Pairing
from poseaudit.types import Instance

SETS = 40


def arms(n: int, rng: np.random.Generator) -> np.ndarray:
    length = rng.uniform(10, 200, n)
    turn = np.radians(rng.uniform(60, 140, n))
    a = np.stack([length, np.zeros(n)], axis=1)
    c = np.stack([length * np.cos(turn), length * np.sin(turn)], axis=1)
    return np.stack([a, np.zeros((n, 2)), c], axis=1) + rng.uniform(0, 500, (n, 1, 2))


def read(truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    first = np.hypot(*(truth[:, 0] - truth[:, 1]).T)
    second = np.hypot(*(truth[:, 2] - truth[:, 1]).T)
    size = np.stack([first, (first + second) / 2, second], axis=1)
    spread = np.hypot(2.0, 0.09 * size)[..., None]
    return truth + rng.normal(size=truth.shape) * spread


def pairing(truth: np.ndarray, predicted: np.ndarray) -> Pairing:
    box, seen = np.array([-1e4, -1e4, 1e4, 1e4]), np.ones(3, bool)
    return Pairing(
        pairs=[
            Pair(f"{i}.jpg", Instance(box, t, seen), Instance(box, p, seen))
            for i, (t, p) in enumerate(zip(truth, predicted, strict=True))
        ]
    )


def main() -> None:
    warnings.simplefilter("ignore")
    print(
        "| readings | check passed | failed | too few to check | "
        "size, 5th / 50th / 95th percentile |"
    )
    print("|---|---|---|---|---|")
    for n in (100, 300, 1000):
        passed, failed, unchecked, sizes = 0, 0, 0, []
        for k in range(SETS):
            rng = np.random.default_rng(1000 * n + k)
            truth = arms(n, rng)
            bent = budget(
                pairing(truth, read(truth, rng)), angle(0, 1, 2), 15, 0.2, seed=k
            ).groups[1]
            passed += bent.passed
            unchecked += not bent.checks
            failed += bool(bent.checks) and not bent.passed
            if bent.smallest_limb_px is not None:
                sizes.append(bent.smallest_limb_px)
        span = "-"
        if sizes:
            low, middle, high = np.percentile(sizes, [5, 50, 95])
            span = f"{low:.0f} / {middle:.0f} / {high:.0f} px"
        print(f"| {n:,} | {passed}/{SETS} | {failed} | {unchecked} | {span} |")


if __name__ == "__main__":
    main()
