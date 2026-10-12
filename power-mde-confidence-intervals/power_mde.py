"""Мощность, MDE и планирование объёма выборки для одновыборочного биномиального теста.

H0: p = p0; H1: p > p0. Входные вероятности задаются долями от 0 до 1.
Основано на идеях лекции Авито «Stat power.ipynb» (ссылка в README).
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n должен быть положительным целым числом")
    return int(n)


def _validate_probability(name: str, value: float, *, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} должен быть конечным числом")
    value = float(value)
    if not (0 < value < 1 if strict else 0 <= value <= 1):
        raise ValueError(f"{name} должен находиться в {'(0, 1)' if strict else '[0, 1]'}")
    return value


def critical_value_right(n: int, p0: float, alpha: float = 0.05) -> int:
    """Правосторонняя критическая область: отвергнуть H0 при Q >= C.

    n — число независимых испытаний Бернулли;
    p0 — вероятность успеха при H0;
    alpha — допустимый уровень ошибки первого рода.
    """
    n = _validate_n(n)
    p0 = _validate_probability("p0", p0)
    alpha = _validate_probability("alpha", alpha, strict=True)

    # вычисляем критическое значение (комментарий из лекции)
    return int(binom.ppf(1 - alpha, n, p0)) + 1


def binomial_power(n: int, p0: float, p_alt: float, alpha: float = 0.05) -> float:
    """Вероятность отвергнуть H0 при истинной вероятности успеха p_alt.

    Это мощность *одновыборочного правостороннего* биномиального критерия,
    а не мощность сравнения двух групп в A/B-тесте.
    """
    n = _validate_n(n)
    p_alt = _validate_probability("p_alt", p_alt)
    critical = critical_value_right(n=n, p0=p0, alpha=alpha)

    # вычисляем мощность по формуле (комментарий из лекции)
    return float(binom.sf(critical - 1, n, p_alt))


def binomial_mde(
    n: int, p0: float, alpha: float = 0.05, target_power: float = 0.8
) -> float:
    """Минимальное *абсолютное* увеличение p над p0, заметное с target_power.

    Пример: p0=0.50 и результат 0.10 означают +10 процентных пунктов,
    а не +10% относительно исходной конверсии.
    """
    n = _validate_n(n)
    p0 = _validate_probability("p0", p0)
    alpha = _validate_probability("alpha", alpha, strict=True)
    target_power = _validate_probability("target_power", target_power, strict=True)

    current_power = binomial_power(n, p0, p0, alpha)
    if current_power >= target_power:
        return 0.0
    if binomial_power(n, p0, 1.0, alpha) < target_power:
        raise ValueError("Невозможно достичь мощности: попробуйте увеличить n или alpha")

    # В лекции MDE подбирается по сетке delta; здесь ищем границу точнее.
    p_alt = brentq(
        lambda p: binomial_power(n, p0, p, alpha) - target_power,
        p0,
        1.0,
        xtol=1e-12,
    )
    return float(p_alt - p0)


def required_sample_size(
    p0: float,
    mde: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    max_n: int = 100_000,
) -> int:
    """Минимальное n, достигающее заданной мощности для эффекта ``mde``.

    Проверяем n по порядку: у точного биномиального критерия мощность
    не обязательно монотонна по n из-за дискретности порога.
    """
    p0 = _validate_probability("p0", p0)
    alpha = _validate_probability("alpha", alpha, strict=True)
    target_power = _validate_probability("target_power", target_power, strict=True)
    max_n = _validate_n(max_n)
    if isinstance(mde, bool) or not isinstance(mde, Real) or not np.isfinite(mde):
        raise ValueError("mde должен быть конечным числом")
    mde = float(mde)
    if not (0 < mde <= 1 - p0):
        raise ValueError("mde должен быть положительным, а p0 + mde <= 1")

    p_alt = p0 + mde
    chunk_size = 1000
    for first_n in range(1, max_n + 1, chunk_size):
        ns = np.arange(first_n, min(first_n + chunk_size, max_n + 1))
        # Критическое значение отдельное для каждого n.
        criticals = binom.ppf(1 - alpha, ns, p0) + 1
        powers = binom.sf(criticals - 1, ns, p_alt)
        found = np.flatnonzero(powers >= target_power)
        if found.size:
            return int(ns[found[0]])
    raise ValueError(f"При n <= {max_n} нужная мощность не достигнута")


# -----------------------------------------------------------------------------
# НЕОБЯЗАТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ (Matplotlib)
# Все функции возвращают фигуру: в Jupyter вызови plt.show(),
# в обычном Python-файле используй figure.savefig("имя.png") и plt.show().
# -----------------------------------------------------------------------------

def plot_power_vs_sample_size(
    p0: float,
    p_alt: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    max_n: int = 300,
):
    """График мощности в зависимости от числа независимых испытаний."""
    import matplotlib.pyplot as plt

    max_n = _validate_n(max_n)
    p0 = _validate_probability("p0", p0)
    p_alt = _validate_probability("p_alt", p_alt)
    alpha = _validate_probability("alpha", alpha, strict=True)
    target_power = _validate_probability("target_power", target_power, strict=True)

    ns = np.arange(1, max_n + 1, dtype=int)
    # Сравниваем критическую область с альтернативным распределением.
    criticals = binom.ppf(1 - alpha, ns, p0) + 1
    powers = binom.sf(criticals - 1, ns, p_alt)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ns, powers, label="Мощность")
    ax.axhline(target_power, linestyle="--", label=f"Цель: {target_power:.0%}")
    ax.set(xlabel="Размер выборки n", ylabel="Мощность",
           title="Мощность и размер выборки", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_power_vs_effect(
    n: int,
    p0: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
):
    """График мощности от прироста p и вертикальная линия MDE (если достижим)."""
    import matplotlib.pyplot as plt

    n = _validate_n(n)
    p0 = _validate_probability("p0", p0)
    alpha = _validate_probability("alpha", alpha, strict=True)
    target_power = _validate_probability("target_power", target_power, strict=True)
    if p0 >= 1:
        raise ValueError("Для графика MDE нужно p0 < 1")

    effects = np.linspace(0, 1 - p0, 251)
    critical = critical_value_right(n, p0, alpha)
    power_values = binom.sf(critical - 1, n, p0 + effects)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(effects * 100, power_values, label="Мощность")
    ax.axhline(target_power, linestyle="--", label=f"Цель: {target_power:.0%}")
    try:
        mde = binomial_mde(n, p0, alpha, target_power)
        ax.axvline(mde * 100, linestyle=":", label=f"MDE: +{mde*100:.2f} п.п.")
    except ValueError:
        ax.text(0.05, 0.1, "Целевая мощность недостижима при этом n",
                transform=ax.transAxes)
    ax.set(xlabel="Абсолютный прирост, процентные пункты", ylabel="Мощность",
           title=f"Мощность и MDE (n={n})", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_binomial_distributions(
    n: int,
    p0: float,
    p_alt: float,
    alpha: float = 0.05,
):
    """Распределения при H0 и H1, критическая область, мощность и ошибка I рода.

    Аналог рисунка из лекции: отмечает, при каком числе успехов
    отклоняется H0. Удобнее всего для относительно небольшого n.
    """
    import matplotlib.pyplot as plt

    n = _validate_n(n)
    p0 = _validate_probability("p0", p0)
    p_alt = _validate_probability("p_alt", p_alt)
    critical = critical_value_right(n, p0, alpha)
    # Отображаем основной диапазон без миллионов почти нулевых столбиков.
    if n > 2000:
        raise ValueError("График распределений удобен при n <= 2000")
    x = np.arange(n + 1)
    under_h0 = binom.pmf(x, n, p0)
    under_h1 = binom.pmf(x, n, p_alt)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.vlines(x, 0, under_h0, linewidth=3, alpha=0.6, label="Распределение при H0")
    ax.vlines(x + 0.12, 0, under_h1, linewidth=3, alpha=0.6, label="Распределение при H1")
    if critical <= n:
        ax.axvspan(critical - 0.5, n + 0.5, alpha=0.12,
                   label=f"Критическая область: Q >= {critical}")
    power = binomial_power(n, p0, p_alt, alpha)
    fpr = float(binom.sf(critical - 1, n, p0))
    ax.set(xlabel="Число успехов Q", ylabel="Вероятность",
           title=f"Два распределения | мощность={power:.1%}, FPR={fpr:.1%}")
    ax.set_xlim(-0.5, n + 0.5)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # ВВОД ДАННЫХ — МОЖНО МЕНЯТЬ И ЗАПУСКАТЬ ФАЙЛ ОТДЕЛЬНО.
    # python power_mde.py
    N = 30                  # Число наблюдений (одна выборка)
    P0 = 0.50               # Вероятность успеха при H0
    P_ALT = 0.60            # Ожидаемая вероятность при альтернативе
    ALPHA = 0.05            # Уровень значимости
    TARGET_POWER = 0.80     # Требуемая мощность
    DESIRED_EFFECT = 0.10   # Желаемая разница в долях (+10 п.п.)
    SHOW_PLOTS = True       # False — только численные результаты
    SAVE_PLOTS = True       # Сохранить графики в PNG рядом с файлом

    print(f"Мощность: {binomial_power(N, P0, P_ALT, ALPHA):.2%}")
    print(f"MDE: +{100 * binomial_mde(N, P0, ALPHA, TARGET_POWER):.2f} п.п.")
    minimum_n = required_sample_size(P0, DESIRED_EFFECT, ALPHA, TARGET_POWER)
    print(f"Минимальное n для +{DESIRED_EFFECT*100:.1f} п.п.: {minimum_n}")

    if SHOW_PLOTS:
        from pathlib import Path
        import matplotlib.pyplot as plt

        figures = [
            ("power_vs_sample_size.png",
             plot_power_vs_sample_size(P0, P_ALT, ALPHA, TARGET_POWER,
                                       max_n=max(N * 8, minimum_n + 20))),
            ("power_vs_effect_mde.png",
             plot_power_vs_effect(N, P0, ALPHA, TARGET_POWER)),
            ("binomial_distributions.png",
             plot_binomial_distributions(N, P0, P_ALT, ALPHA)),
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
