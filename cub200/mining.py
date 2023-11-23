import os
import sys

from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from functools import lru_cache
from os.path import join
import hydra
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


from shared import get_output_dir, train_mln
from compiler import ConstraintCompiler
from detectors import (
    get_predictions,
    MLNDetector,
)
from mln import MLN

log = logging.getLogger(__name__)

import pandas as pd


def generate_expressions(depth, propositions):
    # Define logical operators (excluding 'not' since it's only applied to propositions)
    operators = [
        # ConstraintCompiler.And,
        # ConstraintCompiler.Or,
        # ConstraintCompiler.Xor,
        ConstraintCompiler.Implies,
    ]

    if depth == 0:
        # Base case: return propositions and their negations
        expressions = [ConstraintCompiler.BinaryVariable(p) for p in propositions]
        expressions += [
            ConstraintCompiler.Not(ConstraintCompiler.BinaryVariable(p))
            for p in propositions
        ]
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
    log.info(f"Total unique expressions: {len(unique_expressions)}\n")

    return unique_expressions


def generate_all_constraints(cfg, domain, depth):
    """ """
    attribute_index_map = {k: v for v, k in enumerate(cfg.attributes)}

    # Initialize the compiler with the attribute-index map
    compiler = ConstraintCompiler(attribute_index_map)

    # Prepare the global namespace with torch and index constants
    global_namespace = {
        "torch": torch,
    }

    # Create a dedicated namespace dictionary
    namespace = {}

    constraints = []

    for att, dom in zip(cfg.attributes, domain):
        if len(dom) > 2:
            for value in dom:
                constraints.append(f"{att}={value}")
                constraints.append(f"not {att}={value}")
        else:
            constraints.append(f"{att}")
            constraints.append(f"not {att}")

    constraints += cfg.constraints[:]

    for input_constraint in constraints:
        # input_constraint = "male -> not makeup"
        constraint_name = (
            input_constraint.replace(" ", "_")
            .replace("->", "implies")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "_eq_")
        )

        generated_code = compiler.compile(constraint_name, input_constraint)

        log.info(generated_code)
        # Execute the generated code within this namespace
        exec(generated_code, global_namespace, namespace)

    return list(namespace.values())


def get_base_performance(cfg):
    return 0.5
    # log.info("Generating base performance metrics")
    # base_perfs = []
    # for seed in tqdm(range(cfg.n_seeds)):
    #     data_in, data_ood, data_train = load_data(cfg, seed)
    #     msp_id = -data_in["class-logits"].max(dim=1).values
    #     msp_ood = -data_ood["class-logits"].max(dim=1).values
    #
    #     # ensemble_in = ensemble_score(data_in, cfg.attributes)
    #     # ensemble_ood = ensemble_score(data_ood, cfg.attributes)
    #     metrics = OODMetrics()
    #     metrics.update(msp_id, torch.ones(msp_id.shape[0]))
    #     metrics.update(msp_ood, -torch.ones(msp_ood.shape[0]))
    #     performance = metrics.compute()["AUROC"]
    #     base_perfs.append(performance)
    # best_perf = sum(base_perfs) / len(base_perfs)
    # log.info(f"Base performance is {best_perf:.2%} AUROC")
    # return best_perf


def get_domain(cfg):
    domain = []
    for att in cfg.attributes:
        if att == "class":
            domain.append(range(200))
        else:
            domain.append(range(2))

    return domain


@hydra.main(config_path="config", config_name="mining.yaml", version_base="1.2")
def main(cfg):
    log.info(f"Output dir: {get_output_dir()}")
    rule_set = []

    stats = []
    domain = get_domain(cfg)
    all_constraints = generate_all_constraints(cfg=cfg, domain=domain, depth=2)

    # ensemble performance
    best_perf = get_base_performance(cfg)

    mln = MLN(constraints=[], domain=domain)

    with logging_redirect_tqdm():
        bar = tqdm(all_constraints)
        for fn in bar:
            log.info(f"Rule: {fn.__name__}")
            current_rules = rule_set[:] + [fn]
            ps = []

            for seed in range(cfg.n_seeds):
                performance = run(cfg, mln, current_rules, seed)
                ps.append(performance)
                log.info(performance)

            performance = sum(ps) / len(ps)

            if performance > best_perf + cfg.min_delta:
                rule_set = current_rules
                log.info("Performance improved, adding rule.")
                best_perf = performance
            else:
                log.info("Rule did not increase performance.")

            log.info(
                f"Performance {best_perf:.2%} at {len(rule_set)} rules: {[fn.__name__ for fn in rule_set]}"
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


def run(cfg, mln, current_rules, seed):
    log.info(f"Loading {seed} ... ")
    data_in, data_ood, data_train = load_data(cfg, seed)
    log.info(f"Training... ")
    mln.constraints = current_rules
    mln.w = nn.Parameter(torch.randn(size=(len(mln.constraints),), dtype=torch.double))
    torch.nn.init.constant_(mln.w.data, 0.0)

    train_data = get_predictions(data_train, cfg.attributes)

    train_mln(cfg, mln, train_data, use_prog=False)
    detector = MLNDetector(mln, attributes=cfg.attributes)

    # detector = MLNEBODetector(
    #     mln, attribute="class", mln_attributes=cfg.attributes
    # )
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
        .replace("_labels", "-labels"): value
        for key, value in data.items()
    }
    data["labels"] = data["class-labels"]
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
