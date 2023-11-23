"""
Script implementing sequential greedy constraint mining

"""

import os
import sys

from pytorch_ood.utils import OODMetrics
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from os.path import join
import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from compiler import (
    ConstraintCompiler,
    Implies,
    And,
    Or,
    Xor,
    Not,
    BinaryVariable,
    CategoricalVariable,
)
from tqdm import tqdm
from mln import MLN
from shared import train_mln
from functools import lru_cache
import pandas as pd
from detectors import MLNEnsembleDetector, get_predictions
import logging

import hydra


def get_output_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


log = logging.getLogger(__name__)


def generate_expressions(depth, propositions):
    # Define logical operators (excluding 'not' since it's only applied to propositions)
    operators = [
        Implies,
        # ConstraintCompiler.And,
        # ConstraintCompiler.Or,
        # ConstraintCompiler.Xor,
    ]

    if isinstance(propositions, dict):
        if depth == 0:
            expressions = []

            for p, values in propositions.items():
                if len(values) == 2:
                    expressions += [BinaryVariable(p), Not(BinaryVariable(p))]
                else:
                    # Base case: return propositions and their negations
                    expressions += [CategoricalVariable(p, v) for v in values]
                    expressions += [Not(CategoricalVariable(p, v)) for v in values]

            return expressions

        else:
            expressions = []
            # Generate all combinations of left and right expressions
            for left_depth in range(0, depth):
                right_depth = depth - 1 - left_depth
                left_expressions = generate_expressions(left_depth, propositions)
                right_expressions = generate_expressions(right_depth, propositions)
                for op in operators:
                    for left in left_expressions:
                        for right in right_expressions:
                            expressions.append(op(left, right))
            return expressions

    else:
        if depth == 0:
            # Base case: return propositions and their negations
            expressions = [BinaryVariable(p) for p in propositions]
            expressions += [Not(BinaryVariable(p)) for p in propositions]
            return expressions
        else:
            expressions = []
            # Generate all combinations of left and right expressions
            for left_depth in range(0, depth):
                right_depth = depth - 1 - left_depth
                left_expressions = generate_expressions(left_depth, propositions)
                right_expressions = generate_expressions(right_depth, propositions)
                for op in operators:
                    for left in left_expressions:
                        for right in right_expressions:
                            expressions.append(op(left, right))
            return expressions


def generate_all_expressions(max_depth, propositions):
    # Generate all expressions up to MAX_DEPTH
    all_expressions = []
    for depth in range(0, max_depth):
        expressions = generate_expressions(depth, propositions)
        all_expressions.extend(expressions)

    # Remove duplicates based on their string representation
    unique_expressions = {repr(expr): expr for expr in all_expressions}

    # Print all unique expressions
    log.info(f"Total unique expressions: {len(unique_expressions)}")

    return unique_expressions


def generate_all_constraints(attributes, depth=2):
    """ """
    # Attribute indices
    attribute_index_map = {k: n for n, k in enumerate(attributes)}

    from dataset import RIVAL10

    constraints = generate_all_expressions(
        depth, propositions=RIVAL10.category_value_map  # attribute_index_map.keys()
    )

    constraints = [c[:] for c in constraints]

    compiler = ConstraintCompiler(
        attribute_index_map, category_value_map=RIVAL10.category_value_map
    )

    # Prepare the global namespace with torch and index constants
    global_namespace = {
        "torch": torch,
    }

    # Create a dedicated namespace dictionary
    namespace = {}

    for input_constraint in constraints:
        # input_constraint = "male -> not makeup"
        constraint_name = (
            input_constraint.replace(" ", "_")
            .replace("=", "_eq_")
            .replace("->", "implies")
            .replace("(", "")
            .replace(")", "")
        )

        log.info(f"Generating constraint {constraint_name}: {input_constraint}")
        generated_code = compiler.compile(constraint_name, input_constraint)

        # log.info(generated_code)
        # Execute the generated code within this namespace
        exec(generated_code, global_namespace, namespace)

    return list(namespace.values())


@hydra.main(config_path="config", config_name="mining.yaml", version_base="1.2")
def main(cfg):
    log.info(f"Output dir: {get_output_dir()}")
    rule_set = []

    stats = []

    all_constraints = generate_all_constraints(cfg.attributes, depth=2)

    best_perf = get_base_performance(cfg)

    mln = MLN(constraints=[], domain=[range(2) for _ in cfg.attributes])

    with logging_redirect_tqdm():
        bar = tqdm(all_constraints)
        for fn in bar:
            log.info(f"Rule: {fn.__name__}")
            current_rules = rule_set[:] + [fn]
            ps = []

            for seed in range(cfg.n_seeds):
                performance = run(cfg, mln, current_rules, seed)
                ps.append(performance)
                # log.info(f"{performance:.3%}")

            performance = sum(ps) / len(ps)
            log.info(f"AUROC: {performance:.3%}")

            if performance > best_perf + cfg.min_delta:
                rule_set = current_rules
                log.info("Performance improved, adding rule.")
                best_perf = performance
            else:
                log.info("Rule did not increase performance.")

            log.info(
                f"Performance {best_perf:.3%} at {len(rule_set)} rules: {[fn.__name__ for fn in rule_set]}"
            )

            stats.append(
                {
                    "constraints:": [fn.__name__ for fn in rule_set],
                    "best_performance": best_perf,
                    "current_performance": performance,
                    "current_performances": ps,
                    "current_constraint": fn.__name__,
                }
            )

            df = pd.DataFrame(stats)
            df.to_csv(join(get_output_dir(), "stats.csv"))


def get_base_performance(cfg):
    return 0.96
    # log.info("Generating base performance metrics")
    # base_perfs = []
    # for seed in tqdm(range(cfg.n_seeds)):
    #     data_in, data_ood, data_train = load_data(cfg, seed)
    #     metrics = OODMetrics()
    #     metrics.update(-ensemble_in, torch.ones(ensemble_in.shape[0]))
    #     metrics.update(-ensemble_ood, -torch.ones(ensemble_ood.shape[0]))
    #     performance = metrics.compute()["AUROC"]
    #     base_perfs.append(performance)
    # best_perf = sum(base_perfs) / len(base_perfs)
    # log.info(f"Base performance is {best_perf:.2%} AUROC")
    # return best_perf


def run(cfg, mln, current_rules, seed):
    # log.info(f"Loading {seed} ... ")
    data_in, data_ood, data_train = load_data(cfg, seed)
    # log.info(f"Training... ")
    mln.constraints = current_rules
    mln.w = nn.Parameter(torch.randn(size=(len(mln.constraints),), dtype=torch.double))
    torch.nn.init.constant_(mln.w.data, 0.0)

    train_data = get_predictions(data_train, cfg.attributes)

    train_mln(cfg, mln, train_data, use_prog=False)

    detector = MLNEnsembleDetector(
        mln, ensemble_attributes=cfg.attributes, mln_attributes=cfg.attributes
    )
    detector.fit(cfg, data_train)

    metrics = detector.evaluate(cfg, data_in, data_ood)

    performance = metrics.compute()["AUROC"]
    return performance


def prep_data(data):

    data = {
        key.lower()
        .replace("-", "_")
        .replace("_logits", "-logits")
        .replace("_features", "-features")
        .replace("_labels", "-labels"): value.half()
        for key, value in data.items()
        if "-logits" in key
    }
    # data["labels"] = data["class-labels"]
    return data


@lru_cache(maxsize=10)
def load_data(cfg, seed):
    data_train = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.mln.train_on}-{seed:05d}.pt"),
        weights_only=False,
    )
    data_train = prep_data(data_train)

    data_in = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.validate_on}-{seed:05d}.pt"),
        weights_only=False,
    )
    data_in = prep_data(data_in)

    data_ood = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.dataset}-{seed:05d}.pt"),
        weights_only=False,
    )
    data_ood = prep_data(data_ood)

    return data_in, data_ood, data_train


if __name__ == "__main__":
    main()
