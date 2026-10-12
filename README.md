# Applied Statistics · Прикладная статистика

Личный учебный репозиторий по прикладной статистике: понятные ноутбуки, в которых меняешь входные значения и получаешь расчёты и графики.

## Разделы

| Тема | Ноутбук |
|---|---|
| **Статистический критерий, p-value и критическая область** (существующая работа) | [Binomial Hypothesis Testing](./statistical%20test%3A%20comparison%20of%20the%20critical%20region%20and%20the%20p-value/binomial_hypothesis_test.ipynb) |
| **Статистическая мощность** | [01_power.ipynb](./power-mde-confidence-intervals/01_power.ipynb) |
| **MDE и размер выборки** | [02_mde_sample_size.ipynb](./power-mde-confidence-intervals/02_mde_sample_size.ipynb) |
| **Доверительные интервалы: перебор и Уилсон** | [03_confidence_intervals.ipynb](./power-mde-confidence-intervals/03_confidence_intervals.ipynb) |

[Инструкция к новым расчётам](./power-mde-confidence-intervals/README.md).

## Как работать

1. Открой нужный `.ipynb` в Jupyter, Google Colab или VS Code.
2. В первой ячейке **ВВОД ДАННЫХ** поменяй значения на свои.
3. Нажми **Run All** — получишь результаты и графики; чтобы скрыть графики, поставь `SHOW_PLOTS = False`.

Расчёты по мощности и MDE относятся к одной биномиальной доле, а не к сравнению двух независимых групп.

Материалы по мотивам [открытой лекции Stat power.ipynb Академии аналитиков Авито](https://github.com/avito-tech/applied_statistics/blob/main/Stat%20power.ipynb). Старый раздел по p-value не изменён.
