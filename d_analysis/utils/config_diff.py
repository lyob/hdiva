"""Compare the saved configs of two or more models, field by field.

Pairs with `config_search.py`: find the runs you care about, then diff them.

    from d_analysis.utils.config_search import ConfigQuery, find_models
    from d_analysis.utils.config_diff import diff_configs, print_diff

    nums = find_models(ConfigQuery(rate_type="kl"))
    print_diff(nums[:2])            # just the fields that differ
    diff_configs([17, 32])          # same thing as {field: {model: value}}

`exclude` leaves fields out of the comparison, so the knob you meant to sweep
stops crowding the table and you can see whether anything else drifted:

    print_diff([17, 34], exclude=["beta_init", "lr"])

Configs are read from the dataset directory `config_search` is pointed at, so
`use_dataset("ellipse")` (or `save_loc="ellipse"` on a single call) switches which
runs are compared:

    from d_analysis.utils.config_search import use_dataset
    use_dataset("ellipse")
    print_diff([17, 32])

From the command line:

    python d_analysis/utils/config_diff.py 17 32 --save-loc ellipse
    python d_analysis/utils/config_diff.py 17 32
    python d_analysis/utils/config_diff.py --where rate_type=kl beta_init=0.1
    python d_analysis/utils/config_diff.py 17 34 --exclude beta_init lr
    python d_analysis/utils/config_diff.py 17 32 --all
"""

import argparse
import json

try:  # imported as utils.config_diff
    from .config_search import (
        MISSING,
        _parse_term,
        as_excluded,
        current_dataset,
        field_values,
        find_models,
        load_configs,
        varying_fields,
    )
except ImportError:  # run directly as a script
    from config_search import (
        MISSING,
        _parse_term,
        as_excluded,
        current_dataset,
        field_values,
        find_models,
        load_configs,
        varying_fields,
    )


def diff_configs(
    model_nums: list[int],
    save_loc: str | None = None,
    configs: dict[int, dict] | None = None,
    show_all: bool = False,
    exclude: list[str] | None = None,
) -> dict[str, dict]:
    """field -> {model num: value}, for the fields that differ across `model_nums`.

    `save_loc` is a dataset name ('ellipse') or a path; `None` uses the dataset set
    by `use_dataset`.

    `show_all=True` keeps the identical fields too. `exclude` names fields to leave
    out of the comparison, e.g. to see whether two runs differ by anything beyond
    the knob you swept:

        diff_configs([17, 34], exclude=["beta_init", "lr"])

    A field that a run predates (so it was never saved) shows as MISSING rather
    than as a null value.
    """
    if configs is None:
        configs = load_configs(save_loc)

    unknown = [n for n in model_nums if n not in configs]
    if unknown:
        raise KeyError(f"no saved config for model(s): {unknown}")
    if len(model_nums) < 2:
        raise ValueError("need at least two models to compare")

    exclude = as_excluded(exclude, configs)
    picker = field_values if show_all else varying_fields
    diff = picker(list(model_nums), configs)
    return {k: v for k, v in diff.items() if k not in exclude}


def _render(value, width: int = 28) -> str:
    text = repr(value) if value is MISSING else json.dumps(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def format_diff(diff: dict[str, dict], model_nums: list[int]) -> str:
    """render the diff as an aligned table, one column per model."""
    if not diff:
        return "configs are identical"

    headers = [f"v{n:03d}" for n in model_nums]
    rows = [[field] + [_render(values[n]) for n in model_nums]
            for field, values in diff.items()]

    widths = [
        max(len(row[i]) for row in [["field"] + headers] + rows)
        for i in range(len(model_nums) + 1)
    ]

    def line(cells):
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths)).rstrip()

    return "\n".join(
        [line(["field"] + headers), line(["-" * w for w in widths])]
        + [line(row) for row in rows]
    )


def print_diff(
    model_nums: list[int],
    save_loc: str | None = None,
    configs: dict[int, dict] | None = None,
    show_all: bool = False,
    exclude: list[str] | None = None,
):
    """print the field-by-field comparison of the given models."""
    diff = diff_configs(model_nums, save_loc, configs, show_all, exclude)
    label = "fields" if show_all else "differing fields"
    # diff_configs already validated these; just echo them
    names = [exclude] if isinstance(exclude, str) else sorted(exclude or ())
    ignored = f", ignoring {names}" if names else ""
    print(f"{len(diff)} {label} across models {list(model_nums)}{ignored}\n")
    print(format_diff(diff, list(model_nums)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_nums", nargs="*", type=int, help="model numbers to compare")
    parser.add_argument(
        "--where",
        nargs="+",
        metavar="KEY=VALUE",
        help="instead of model numbers, compare every model matching these search terms",
    )
    parser.add_argument(
        "--save-loc",
        default=None,
        metavar="DATASET_OR_PATH",
        help=f"dataset name (default {current_dataset()}) or a path to a weights dir",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="FIELD",
        help="field names to leave out of the comparison, e.g. --exclude beta_init lr",
    )
    parser.add_argument(
        "--all", action="store_true", help="show identical fields as well as differing ones"
    )
    args = parser.parse_args()

    configs = load_configs(args.save_loc)
    model_nums = list(args.model_nums)

    if args.where:
        terms = dict(_parse_term(t) for t in args.where)
        matched = find_models(terms, configs=configs)
        print(f"{len(matched)} models match {terms}: {matched}\n")
        model_nums += [n for n in matched if n not in model_nums]

    if len(model_nums) < 2:
        parser.error("give at least two model numbers (or a --where matching two or more)")

    print_diff(model_nums, configs=configs, show_all=args.all, exclude=args.exclude)


if __name__ == "__main__":
    main()
