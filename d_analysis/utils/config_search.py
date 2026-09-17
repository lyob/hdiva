"""Search the saved model configs for the runs matching a set of field values.

Configs are written by `d_analysis/diva_sami_comparison.py` as
`sami_config_v{model_num:03d}.json` inside `c_training/local_weights/{dataset_name}`.

Search terms can be given either as a dict or as a `ConfigQuery` dataclass
(auto-generated from `Config`, every field defaulting to `None` = "don't care"):

    from d_analysis.utils.config_search import ConfigQuery, find_models

    find_models({"rate_type": "kl", "beta_init": 0.1})
    find_models(ConfigQuery(rate_type="kl", freeze_denoiser=True))

Values are compared exactly (lists/tuples are normalised, since json turns the
tuple fields into lists). A value can also be a callable predicate:

    find_models({"beta_init": lambda b: b >= 1e-2})

`exclude` drops fields from the comparison, which is what you want when the query
is a whole config and only a few fields are allowed to vary:

    configs = load_configs()
    find_models(configs[17], exclude=["beta_init", "lr"])  # runs like v017, betas free

Note the query there is the *saved* dict, not `Config.from_dict(...)`: rehydrating
back-fills defaults for fields that predate the saved run (freeze_denoiser,
pretrained_denoiser), and a field the config never saved never matches — so a
rehydrated Config needs those names in `exclude` too.

Runs are grouped by dataset, one directory per dataset under `c_training/local_weights`.
`save_loc` therefore takes a bare dataset name as well as a path, and `use_dataset`
switches the default for every later call (including those in `config_diff`):

    from d_analysis.utils.config_search import use_dataset, available_datasets

    available_datasets()        # ['ellipse', 'ring']
    use_dataset("ellipse")      # everything below now reads the ellipse runs
    find_models({"rate_type": "kl"})
    find_models({"rate_type": "kl"}, save_loc="ring")   # just this call

From the command line:

    python d_analysis/utils/config_search.py rate_type=kl beta_init=0.1
    python d_analysis/utils/config_search.py rate_type=kl --save-loc ellipse
    python d_analysis/utils/config_search.py rate_type=kl --exclude beta_final
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, fields, is_dataclass, make_dataclass
from typing import Any, Optional

# repo root, resolved from this file so imports work from any cwd
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from b_models.configs.sami_simple_config import Config

WEIGHTS_ROOT = f"{PROJECT_DIR}/c_training/local_weights"
CONFIG_PATTERN = r"sami_config_v(\d+)\.json"

# which dataset's weights/configs to read when a call doesn't name one; set it
# once per session with `use_dataset("ellipse")` rather than passing save_loc
# to every call.
DEFAULT_DATASET = "ring"


def use_dataset(dataset_name: str) -> str:
    """point every later call at `{WEIGHTS_ROOT}/{dataset_name}`; returns the path."""
    global DEFAULT_DATASET
    DEFAULT_DATASET = dataset_name
    return resolve_save_loc()


def current_dataset() -> str:
    """the dataset name that an unset `save_loc` currently resolves to."""
    return DEFAULT_DATASET


def available_datasets() -> list[str]:
    """dataset names that have a directory under `WEIGHTS_ROOT`."""
    if not os.path.isdir(WEIGHTS_ROOT):
        return []
    return sorted(
        d for d in os.listdir(WEIGHTS_ROOT) if os.path.isdir(f"{WEIGHTS_ROOT}/{d}")
    )


def resolve_save_loc(save_loc: str | None = None) -> str:
    """a dataset name ('ellipse') -> its weights dir; a path is passed through.

    `None` means the current `DEFAULT_DATASET`.
    """
    if save_loc is None:
        save_loc = DEFAULT_DATASET
    if os.sep in save_loc or os.path.isdir(save_loc):
        return save_loc  # already a path
    return f"{WEIGHTS_ROOT}/{save_loc}"


# a partial view of Config: same fields, all optional, None means "not searched on"
ConfigQuery = make_dataclass(
    "ConfigQuery",
    [(f.name, Optional[Any], None) for f in fields(Config)],
    namespace={
        "__doc__": "Search terms for `find_models`; unset (None) fields are ignored."
    },
)


class _Missing:
    """a field that the saved config does not contain at all (vs. a saved null)."""

    def __repr__(self):
        return "<missing>"


MISSING = _Missing()


def _normalize(value):
    """json round-trips tuples as lists, so compare both as lists."""
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _as_terms(query) -> dict:
    """accept a dict, a ConfigQuery, or any dataclass; drop the unset fields."""
    if is_dataclass(query):
        query = asdict(query)
    if not isinstance(query, dict):
        raise TypeError(f"query must be a dict or dataclass, got {type(query).__name__}")
    return {k: v for k, v in query.items() if v is not None}


def load_configs(save_loc: str | None = None) -> dict[int, dict]:
    """map model number -> saved config dict, for every versioned config in `save_loc`.

    `save_loc` is a dataset name ('ellipse') or a path; `None` uses DEFAULT_DATASET.
    """
    save_loc = resolve_save_loc(save_loc)
    if not os.path.isdir(save_loc):
        raise FileNotFoundError(
            f"no config directory at {save_loc}; "
            f"datasets with saved weights: {available_datasets()}"
        )

    configs = {}
    for fname in os.listdir(save_loc):
        m = re.fullmatch(CONFIG_PATTERN, fname)
        if m is None:
            continue
        with open(f"{save_loc}/{fname}") as f:
            configs[int(m.group(1))] = json.load(f)
    return dict(sorted(configs.items()))


def _matches(config: dict, terms: dict) -> bool:
    for key, wanted in terms.items():
        if key not in config:
            return False
        found = config[key]
        if callable(wanted):
            if not wanted(found):
                return False
        elif _normalize(found) != _normalize(wanted):
            return False
    return True


def known_fields(configs: dict[int, dict] | None = None) -> set[str]:
    """every field name we recognise: the current Config, plus whatever the saved
    configs contain (older runs carry fields that Config has since dropped)."""
    known = {f.name for f in fields(Config)}
    if configs:
        known = known.union(*(c.keys() for c in configs.values()))
    return known


def as_excluded(exclude, configs: dict[int, dict] | None = None, also_check=()) -> set[str]:
    """normalise an `exclude` argument to a set, rejecting typo'd field names."""
    if isinstance(exclude, str):  # a bare string would iterate as characters
        exclude = [exclude]
    exclude = set(exclude or ())

    unknown = (exclude | set(also_check)) - known_fields(configs)
    if unknown:
        raise KeyError(f"unknown config field(s): {sorted(unknown)}")
    return exclude


def find_models(
    query,
    save_loc: str | None = None,
    configs: dict[int, dict] | None = None,
    exclude: list[str] | None = None,
) -> list[int]:
    """model numbers whose saved config matches every search term in `query`.

    `save_loc` is a dataset name ('ellipse') or a path; `None` uses DEFAULT_DATASET.

    `exclude` names fields to ignore when comparing, e.g. to take a whole saved
    config as the query and find the runs that agree with it apart from a sweep:

        find_models(configs[17], exclude=["beta_init", "lr"])
    """
    terms = _as_terms(query)
    if configs is None:
        configs = load_configs(save_loc)

    exclude = as_excluded(exclude, configs, also_check=terms)
    terms = {k: v for k, v in terms.items() if k not in exclude}
    return [num for num, config in configs.items() if _matches(config, terms)]


def field_values(model_nums: list[int], configs: dict[int, dict]) -> dict[str, dict]:
    """field -> {model num: value} over the given models, MISSING where unsaved."""
    if not model_nums:
        return {}
    keys = set().union(*(configs[n].keys() for n in model_nums))
    return {
        key: {n: _normalize(configs[n][key]) if key in configs[n] else MISSING
              for n in model_nums}
        for key in sorted(keys)
    }


def varying_fields(model_nums: list[int], configs: dict[int, dict]) -> dict[str, dict]:
    """for the given models, the fields that are not identical across all of them."""
    varying = {}
    for key, values in field_values(model_nums, configs).items():
        first, *rest = values.values()
        if any(v != first for v in rest):
            varying[key] = values
    return varying


def _parse_term(term: str) -> tuple[str, Any]:
    if "=" not in term:
        raise argparse.ArgumentTypeError(f"expected key=value, got {term!r}")
    key, _, raw = term.partition("=")
    try:
        value = json.loads(raw)  # numbers, bools, null, lists
    except json.JSONDecodeError:
        value = raw  # plain string, e.g. rate_type=kl
    return key.strip(), value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "terms", nargs="*", help="search terms as key=value, e.g. rate_type=kl"
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
        help="field names to ignore when comparing, e.g. --exclude beta_final",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="also list the config fields that differ across the matched models",
    )
    args = parser.parse_args()

    terms = dict(_parse_term(t) for t in args.terms)
    configs = load_configs(args.save_loc)
    matches = find_models(terms, configs=configs, exclude=args.exclude)

    if args.exclude:
        terms = {k: v for k, v in terms.items() if k not in set(args.exclude)}
    print(f"{len(matches)}/{len(configs)} models match {terms}")
    print(matches)

    if args.show_diff:
        for key, values in varying_fields(matches, configs).items():
            print(f"\n{key}:")
            for num, value in values.items():
                print(f"  v{num:03d}: {value}")


if __name__ == "__main__":
    main()
