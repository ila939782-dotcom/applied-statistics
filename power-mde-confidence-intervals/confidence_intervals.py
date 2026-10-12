"""Доверительные интервалы для вероятности успеха в испытаниях Бернулли.

Два способа: перебор (инверсия биномиального критерия) и интервал Уилсона.
``side='two-sided'`` — двусторонний интервал;
``side='lower'`` — односторонняя нижняя доверительная граница.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint


def _validate_inputs(successes: int, n: int, alpha: float, side: str) -> None:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n должен быть положительным целым числом")
    if isinstance(successes, bool) or not isinstance(successes, Integral):
        raise ValueError("successes должно быть целым числом")
    if not 0 <= successes <= n:
        raise ValueError("Требуется 0 <= successes <= n")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha):
        raise ValueError("alpha должен быть конечным числом")
    if not 0 < alpha < 1:
        raise ValueError("Требуется 0 < alpha < 1")
    if side not in ("two-sided", "lower"):
        raise ValueError("side должен быть 'two-sided' или 'lower'")
    if side == "lower" and alpha >= 0.5:
        raise ValueError("Для нижней границы используйте alpha < 0.5")


def ci_by_inversion(
    successes: int,
    n: int,
    alpha: float = 0.05,
    grid_step: float = 0.001,
    side: str = "two-sided",
) -> tuple[float, float]:
    """Интервал перебором p0, при которых H0 не отвергается.

    Это *приближение по сетке* инверсии точного биномиального критерия;
    точность границ зависит от ``grid_step`` (по умолчанию 0.001).
    """
    _validate_inputs(successes, n, alpha, side)
    if isinstance(grid_step, bool) or not isinstance(grid_step, Real) or not np.isfinite(grid_step):
        raise ValueError("grid_step должен быть конечным числом")
    if not 0 < grid_step <= 0.1:
        raise ValueError("grid_step должен быть > 0 и <= 0.1")

    # Проверяем гипотезу при каждом возможном p0, включая 0 и 1.
    grid_points = int(np.ceil(1.0 / grid_step))
    if grid_points > 1_000_000:
        raise ValueError("Слишком мелкая сетка; увеличьте grid_step")
    candidate_p0 = np.linspace(0.0, 1.0, grid_points + 1)

    if side == "two-sided":
        # аналогично одностороннему критерию (комментарий из лекции)
        c2 = binom.ppf(1 - alpha / 2, n, candidate_p0) + 1
        # по выкладкам выше (комментарий из лекции)
        c1 = binom.ppf(alpha / 2, n, candidate_p0) - 1
        accepted = (successes > c1) & (successes < c2)
    else:
        c = binom.ppf(1 - alpha, n, candidate_p0) + 1
        accepted = successes < c

    feasible = candidate_p0[accepted]
    if not feasible.size:
        raise ValueError("Сетка слишком грубая для этого n; уменьшите grid_step")
    left, right = float(feasible[0]), float(feasible[-1])
    return (left, 1.0) if side == "lower" else (left, right)


def ci_wilson(
    successes: int, n: int, alpha: float = 0.05, side: str = "two-sided"
) -> tuple[float, float]:
    """Быстрый приближённый интервал Уилсона для биномиальной доли.

    Для односторонней нижней границы используется z_(1-alpha), то есть
    нижняя граница двустороннего Wilson-интервала уровня 1-2*alpha.
    """
    _validate_inputs(successes, n, alpha, side)
    lo, hi = proportion_confint(
        count=int(successes),  # количество успехов (комментарий из лекции)
        nobs=int(n),           # размер выборки (комментарий из лекции)
        alpha=float(2 * alpha if side == "lower" else alpha),
        method="wilson",
    )
    return (float(lo), 1.0) if side == "lower" else (float(lo), float(hi))


# -----------------------------------------------------------------------------
# НЕОБЯЗАТЕЛЬНЫЕ ГРАФИКИ. plt.show() — в Jupyter, fig.savefig() — в PNG.
# -----------------------------------------------------------------------------

def plot_confidence_intervals(
    successes: int,
    n: int,
    alpha: float = 0.05,
    grid_step: float = 0.001,
):
    """Сравнение двусторонних интервалов и нижних границ двух методов."""
    import matplotlib.pyplot as plt

    grid_ci = ci_by_inversion(successes, n, alpha, grid_step)
    wilson_ci = ci_wilson(successes, n, alpha)
    grid_lower = ci_by_inversion(successes, n, alpha, grid_step, side="lower")[0]
    wilson_lower = ci_wilson(successes, n, alpha, side="lower")[0]
    observed = successes / n

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for level, ci, label in [(3, grid_ci, "Перебор (двусторонний)"),
                              (2, wilson_ci, "Уилсон (двусторонний)"),
                              (1, (grid_lower, 1.0), "Перебор (нижняя граница)"),
                              (0, (wilson_lower, 1.0), "Уилсон (нижняя граница)")]:
        ax.plot(ci, [level, level], "o-", label=label)
    ax.axvline(observed, linestyle="--", label=f"Наблюдаемая доля: {observed:.1%}")
    ax.set(xlabel="Вероятность успеха", title=f"Доверительные интервалы: {1-alpha:.0%}")
    ax.set_yticks([0, 1, 2, 3], ["Уилсон, нижняя", "Перебор, нижняя",
                                   "Уилсон", "Перебор"])
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_wilson_coverage(
    true_p: float = 0.5,
    alpha: float = 0.05,
    max_n: int = 120,
    step_n: int = 5,
):
    """Точное покрытие интервала Уилсона при разных размерах выборки.

    Вместо долгой симуляции суммируем вероятности всех возможных Q
    для заданного истинного p (биномиальное распределение).
    """
    import matplotlib.pyplot as plt

    if isinstance(true_p, bool) or not isinstance(true_p, Real) or not 0 <= true_p <= 1:
        raise ValueError("true_p должен быть в [0, 1]")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha должен быть в (0, 1)")
    if isinstance(max_n, bool) or not isinstance(max_n, Integral) or max_n < 1:
        raise ValueError("max_n должен быть положительным целым числом")
    if isinstance(step_n, bool) or not isinstance(step_n, Integral) or step_n < 1:
        raise ValueError("step_n должен быть положительным целым числом")

    ns = list(range(1, int(max_n) + 1, int(step_n)))
    if ns[-1] != max_n:
        ns.append(int(max_n))
    coverages = []
    for n in ns:
        outcomes = np.arange(n + 1)
        lo, hi = proportion_confint(outcomes, nobs=n, alpha=alpha, method="wilson")
        includes_true = (lo <= true_p) & (true_p <= hi)
        coverages.append(float(binom.pmf(outcomes, n, true_p) @ includes_true))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ns, coverages, marker="o", markersize=3, label="Реальное покрытие")
    ax.axhline(1 - alpha, linestyle="--", label=f"Номинальное: {1-alpha:.0%}")
    ax.set(xlabel="Размер выборки n", ylabel="Вероятность покрытия",
           title="Покрытие доверительного интервала Уилсона", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # ВВОД ДАННЫХ — МОЖНО МЕНЯТЬ И ЗАПУСКАТЬ ФАЙЛ ОТДЕЛЬНО.
    # python confidence_intervals.py
    N = 30                # Всего испытаний
    SUCCESSES = 19        # Сколько успехов
    ALPHA = 0.05          # 95% уровень доверия
    GRID_STEP = 0.001     # Шаг перебора гипотез
    SHOW_PLOTS = True     # False — только числа
    SAVE_PLOTS = True     # Сохранить графики в PNG рядом с файлом

    for name, fn in [("Перебор", ci_by_inversion), ("Уилсон", ci_wilson)]:
        kwargs = {"grid_step": GRID_STEP} if name == "Перебор" else {}
        lo, hi = fn(SUCCESSES, N, ALPHA, **kwargs)
        print(f"{name}: {lo:.2%} — {hi:.2%}")
        lower, _ = fn(SUCCESSES, N, ALPHA, side="lower", **kwargs)
        print(f"{name} (нижняя граница): p >= {lower:.2%}")

    if SHOW_PLOTS:
        from pathlib import Path
        import matplotlib.pyplot as plt

        figures = [
            ("ci_comparison.png", plot_confidence_intervals(SUCCESSES, N, ALPHA, GRID_STEP)),
            ("wilson_coverage.png", plot_wilson_coverage(true_p=SUCCESSES / N, alpha=ALPHA)),
        ]
        for filename, fig in figures:
            if SAVE_PLOTS:
                path = Path(__file__).resolve().parent / filename
                fig.savefig(path, dpi=140, bbox_inches="tight")
                print(f"Сохранён график: {path}")
        # В терминале без графического интерфейса окно не откроется:
        # PNG уже сохранены; избегаем предупреждения про non-interactive backend.
        if "agg" in plt.get_backend().lower():
            print("Графический интерфейс недоступен; открой сохранённые PNG.")
            plt.close("all")
        else:
            plt.show()
