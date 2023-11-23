import copy
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from os.path import join
from typing import List, Dict
import multiprocessing as mp
import pickle
import numpy as np
import socket

import pandas as pd
import torch

from torch import nn

from compiler import (
    ConstraintCompiler,
    Not,
    Node,
    BinaryVariable,
    BinaryOperator,
    UnaryOperator,
    CategoricalVariable,
    Xor,
    And,
    Or,
    Implies,
    Operator,
)
from detectors import get_predictions, MLNEnsembleDetector
from mln import MLN
from shared import (
    train_mln,
    get_output_dir,
    Timer,
    train_mln_pseudo,
    file_lock,
    get_machine_lock_file,
)

try:
    from mpi4py import MPI
except ImportError:
    print(f"mpi not installed, there might be problems")

log = logging.getLogger(__name__)


class Individual:
    """
    Encapsulates a candidate solution: a set of constraint strings.
    """

    def __init__(self, constraints=None):
        """
        constraints: List[str], each a textual representation of a constraint
        """
        self.constraints: List[Node] = []

        constraints = constraints or []

        self._sig = None

        for constraint in constraints:
            if not isinstance(constraint, Node):
                raise ValueError(f"constraint {constraint} is not a Node")
            else:
                self.constraints.append(constraint)

        self._update_sig()

    def _update_sig(self):
        sorted_strs = sorted([str(s) for s in self.constraints])
        self._sig = ";".join(sorted_strs)

    def copy(self) -> "Individual":
        return copy.deepcopy(self)

    def has_constraint(self, root: Node) -> bool:
        for c in self.constraints:
            if str(c) == str(root):
                return True

        return False

    def add_constraint(self, root: Node) -> bool:
        if root is None:
            log.warning(f"tried to add none constraint")
            return False

        if self.has_constraint(root):
            log.debug(f"constraint {root} already added")
            return False

        self.constraints.append(root)
        self._update_sig()
        return True

    def __len__(self):
        return len(self.constraints)

    def __repr__(self):
        return f"Individual<{self._sig}>"

    def signature(self) -> str:
        """
        Create a *canonical* representation of the constraint set.
        We can just sort the strings and join them.
        """
        return self._sig


class Selector(ABC):
    """
    The selector selects individuals
    """

    @abstractmethod
    def select(self, individuals: List[Individual], metrics) -> List[Individual]:
        pass


class ObjectiveFunctionSelector(Selector):
    """
    Selects based on value of the given metric, which should be the fitness
    """

    def __init__(self, survival_rate, metric="fitness"):
        self.survival_rate = survival_rate
        self.metric = metric

    def select(
        self, individuals: List[Individual], results: List[Dict]
    ) -> List[Individual]:
        fitnesses = [r[self.metric] for r in results]

        ranked_pop = sorted(
            zip(individuals, fitnesses), key=lambda x: x[1], reverse=True
        )

        survivors = [
            x[0]
            for x in ranked_pop[: (len(individuals) // int(1 / self.survival_rate))]
        ]

        # print top 10
        for n, (ind, fit) in enumerate(ranked_pop[:10]):
            log.info(f"Top {n}: {fit:.05f} -> {ind}")

        return survivors


class ParetoFrontSelector(Selector):
    """
    Selects individuals based on dominance
    """

    def __init__(
        self,
        survival_rate,
        metrics: List[str],
        directions: List[str],
        use_crowding_distance=True,
    ):
        self.metrics = metrics
        self.directions = directions
        self.survival_rate = survival_rate
        self.use_crowding_distance = use_crowding_distance

    def select(
        self, individuals: List[Individual], results: List[Dict]
    ) -> List[Individual]:
        df = pd.DataFrame(results)

        fronts = self.vectorized_non_dominated_sort(
            df, self.metrics, {m: d for m, d in zip(self.metrics, self.directions)}
        )

        survivors = []

        target = len(individuals) // int(1 / self.survival_rate)

        if self.use_crowding_distance:
            for front in fronts:
                rest_size = target - len(survivors)
                if len(front) > rest_size:
                    front_df = df.iloc[front]

                    distances = self.compute_crowding_distance_for_front(
                        front_df,
                        self.metrics,
                        {m: d for m, d in zip(self.metrics, self.directions)},
                    )

                    idx = distances.sort_values(ascending=False).index[:rest_size]
                    survivors.extend([individuals[i] for i in idx])
                else:
                    survivors.extend([individuals[i] for i in front])
        else:
            current_front = 0

            while True:
                for ind in fronts[current_front]:
                    survivors.append(individuals[ind])
                    if len(survivors) >= target:
                        return survivors

                current_front += 1

        return survivors

    @staticmethod
    def compute_crowding_distance_for_front(front_df, metrics, directions):
        """
        Compute NSGA-II style crowding distances for one non-dominated front.

        Parameters
        ----------
        front_df : pd.DataFrame
            DataFrame containing only the solutions in a single front.
        metrics : list of str
            The objectives we used in the Pareto sort, e.g. ["performance","complexity"].
        directions : dict
            Maps each metric to "min" or "max".

        Returns
        -------
        crowding_dist : pd.Series
            Crowding distance for each row in 'front_df', same index as 'front_df'.
        """
        if front_df.shape[0] <= 2:
            # With 0, 1, or 2 solutions, we can just assign infinity (or 0 if you prefer)
            # because they won't be crowded relative to one another.
            # Typically, with 2 solutions, both get 'inf' so they're preserved.
            dist = np.full(front_df.shape[0], fill_value=np.inf)
            return pd.Series(dist, index=front_df.index)

        # Initialize all distances to 0
        dist = np.zeros(front_df.shape[0], dtype=float)

        for m in metrics:
            # Sort by objective: ascending if "min", descending if "max"
            ascending = directions[m] == "min"
            sorted_idx = front_df[m].sort_values(ascending=ascending).index
            sorted_vals = front_df.loc[sorted_idx, m].values

            # f_min, f_max for normalization
            f_min, f_max = sorted_vals[0], sorted_vals[-1]
            span = f_max - f_min
            if span == 0:
                # If all solutions have the same value for this metric,
                # there's no crowding contribution from this objective.
                # Continue to next objective.
                continue

            # Boundary points get infinite distance
            dist[front_df.index.get_loc(sorted_idx[0])] = np.inf
            dist[front_df.index.get_loc(sorted_idx[-1])] = np.inf

            # For interior points, add normalized distance
            for i in range(1, len(sorted_idx) - 1):
                idx_i = sorted_idx[i]
                idx_im1 = sorted_idx[i - 1]
                idx_ip1 = sorted_idx[i + 1]
                dist_i = (front_df.at[idx_ip1, m] - front_df.at[idx_im1, m]) / span
                # Accumulate in our distance array
                dist[front_df.index.get_loc(idx_i)] += dist_i

        # Return as a Series
        return pd.Series(dist, index=front_df.index)

    @staticmethod
    def vectorized_non_dominated_sort(df, metrics, directions):
        """
        Vectorized approach to build all Pareto fronts for multi-objective data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns in 'metrics'.
        metrics : list of str
            Objective columns, e.g. ["auroc", "complexity", "aupr-id", "aupr-ood"].
        directions : dict
            Maps each metric -> "min" or "max".

        Returns
        -------
        fronts : list of list
            Each element is a list of row indices (from df.index) belonging
            to that Pareto front, in ascending order (first front = best).
        """
        # 1) Convert "max" objectives into "min" by negating
        #    Build an array of shape (N, M).
        data_arrays = []
        for m in metrics:
            if directions[m] == "max":
                data_arrays.append(-df[m].to_numpy())
            else:
                data_arrays.append(df[m].to_numpy())

        points = np.column_stack(data_arrays)  # shape = (N, M)

        # 2) Build a NxN matrix to check domination
        #    For large N, watch out for memory usage: NxN can be big.
        #    We compare all pairs (i, j) via broadcasting:
        #    "i <= j for all m" => (points[i] <= points[j]).all(axis=-1)
        #    "i < j for at least one m" => (points[i] < points[j]).any(axis=-1)

        # Expand dimensions to broadcast:
        # points[:, None, :] => shape (N, 1, M)
        # points[None, :, :] => shape (1, N, M)
        less_equal = points[:, None, :] <= points[None, :, :]  # shape (N, N, M)
        strictly_less = points[:, None, :] < points[None, :, :]

        # i_dominates_j is True if row i is <= row j in all metrics
        # and < in at least one metric
        i_dominates_j = less_equal.all(axis=2) & strictly_less.any(axis=2)

        # We'll store this in a 2D boolean array
        # i_dominates_j[i, j] = True if i dominates j
        # (No "elif" needed because it's vectorized.)
        dominates_matrix = i_dominates_j

        # 3) Perform a fast non-dominated sort using this precomputed matrix
        N = len(df)
        # S[i] = list of j that i dominates
        S = [[] for _ in range(N)]
        # n[i] = how many j dominate i
        n = [0] * N

        # We just read the NxN matrix: if i_dominates_j[i, j] is True, then i dominates j
        # => j is dominated by i => n[j] += 1
        # => S[i].append(j)
        for i in range(N):
            # row i
            row_i = dominates_matrix[i, :]  # shape (N,)
            # i dominates all j where row_i[j] == True
            dominated_js = np.where(row_i)[0]
            for j in dominated_js:
                S[i].append(j)
                n[j] += 1
        # The rest is the usual "peeling" approach
        indices = df.index.to_list()

        front = [i for i in range(N) if n[i] == 0]
        fronts = []
        while front:
            next_front = []
            fronts.append([indices[x] for x in front])

            for i in front:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        next_front.append(j)
            front = next_front

        return fronts


class MutationOperator(ABC):
    """
    Interface for mutation operators.
    """

    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        pass


class CrossoverOperator(ABC):
    """
    Interface for crossover operators.
    """

    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        pass


class Fitness(ABC):
    @abstractmethod
    def fitness(self, individual: Individual) -> dict:
        """
        Should return a dict with the key "fitness"
        """
        pass


class SimpleCrossover(CrossoverOperator):
    """
    Joins half the parent's constraints with half the other's,
    ensuring we take at least 1 constraint from each.
    """

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        c1 = parent1.copy()
        c2 = parent2.copy()
        random.shuffle(c1.constraints)
        random.shuffle(c2.constraints)

        # Take at least 1 from each
        half1_count = max(1, len(c1) // 2)
        half2_count = max(1, len(c2) // 2)

        half1 = c1.constraints[:half1_count]
        half2 = c2.constraints[:half2_count]

        return Individual(half1 + half2)


class RandomSubsetCrossover(CrossoverOperator):
    """
    Takes random subsets from each parent to form a child.
    """

    def __init__(self, prevent_empty=True):
        self.prevent_empty = prevent_empty

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual()
        for constraint in parent1.constraints:
            if random.random() < 0.5:
                child.add_constraint(copy.deepcopy(constraint))
        for constraint in parent2.constraints:
            if random.random() < 0.5:
                child.add_constraint(copy.deepcopy(constraint))

        if self.prevent_empty and len(child.constraints) == 0:
            # if empty, we just add one constraint
            child.add_constraint(
                copy.deepcopy(random.choice(parent1.constraints + parent2.constraints))
            )

        return child


class FitnessAUROC(Fitness):
    def __init__(self, compiler: ConstraintCompiler, cfg, domain, alpha):
        self.compiler = compiler
        self.cfg = cfg
        self.alpha = alpha

        self.domain = domain
        self.mln = MLN(constraints=[], domain=self.domain)

    def fitness(self, individual: Individual) -> dict:
        """Example worker function."""

        constraints = self.compile_constraints(individual)

        r = defaultdict(list)
        import psutil

        process = psutil.Process()

        try:
            for seed in range(self.cfg.n_seeds):
                mbytes_used = process.memory_info().rss / 1024 / 1024
                log.debug(f"Loading data for seed {seed} - {mbytes_used}M")
                data_in, data_ood, data_train = self.load_data(
                    self.cfg, seed, filter_att=self.cfg.attributes
                )

                mbytes_used = process.memory_info().rss / 1024 / 1024
                log.debug(f"Training on seed {seed} - {mbytes_used}M")
                detector = self.create_detector(self.cfg, constraints, data_train)

                mbytes_used = process.memory_info().rss / 1024 / 1024
                log.debug(f"Evaluating on {seed} - {mbytes_used}M")
                metrics = detector.evaluate(self.cfg, data_in, data_ood)

                for metric, value in metrics.compute().items():
                    r[metric].append(value)

        except ValueError as e:
            import gc

            gc.collect()

            return {
                "complexity": None,
                "performance": None,
                "performances": None,
                "fitness": 0,
            }

        except Exception as e:
            log.exception(e)
            return {
                "complexity": None,
                "performance": None,
                "performances": None,
                "fitness": -1,
            }

        fitness = self._compute_fitness(individual, r)
        complexity = sum([c.size() for c in individual.constraints])

        return {
            "complexity": complexity,
            "auroc": np.array(r["AUROC"]).mean(),
            "aupr-id": np.array(r["AUPR-IN"]).mean(),
            "aupr-ood": np.array(r["AUPR-OUT"]).mean(),
            "fpr95": np.array(r["FPR95TPR"]).mean(),
            "fitness": fitness,
        }

    def _compute_fitness(self, individual, metrics):
        performance = sum(metrics["AUROC"]) / len(metrics["AUROC"])
        complexity = sum([c.size() for c in individual.constraints])
        fitness = performance - self.alpha * complexity
        return fitness

    def create_detector(self, cfg, constraints, data_train):
        self.mln.constraints = constraints
        self.mln.w = nn.Parameter(
            torch.randn(size=(len(self.mln.constraints),), dtype=torch.double)
        )
        torch.nn.init.constant_(self.mln.w.data, cfg.mln.init)
        train_data = get_predictions(data_train, self.cfg.attributes)

        if cfg.mnl.loss == "pseudo":
            train_mln_pseudo(self.cfg, self.mln, train_data, use_prog=cfg.mln.progress)
        elif cfg.mln.loss == "likelihood":
            train_mln(self.cfg, self.mln, train_data, use_prog=cfg.mln.progress)
        else:
            raise ValueError

        detector = MLNEnsembleDetector(
            self.mln,
            ensemble_attributes=self.cfg.attributes,
            mln_attributes=self.cfg.attributes,
        )
        detector.fit(self.cfg, data_train)
        return detector

    def compile_constraints(self, individual):
        # Prepare the global namespace with torch and index constants
        global_namespace = {
            "torch": torch,
        }
        # Create a dedicated namespace dictionary
        namespace = {}
        for constraint in individual.constraints:
            constraint = str(constraint)
            # input_constraint = "male -> not makeup"
            constraint_name = "_" + constraint.replace(" ", "_").replace(
                "->", "implies"
            ).replace("(", "").replace(")", "").replace("=", "_eq_")

            generated_code = self.compiler.compile(constraint_name, constraint)

            log.debug(generated_code)
            # Execute the generated code within this namespace
            exec(generated_code, global_namespace, namespace)
        rules = list(namespace.values())
        return rules

    @lru_cache(maxsize=10)
    # prevent multiple processes from loading at the same time
    @file_lock(get_machine_lock_file())
    def load_data(self, cfg, seed, filter_att=None, to_half=False):

        def prep(x):
            return (
                x.lower()
                .replace("-", "_")
                .replace("(", "_")
                .replace(")", "_")
                .replace("_logits", "-logits")
                .replace("_features", "-features")
            )

        try:
            data_train = torch.load(
                join(cfg.paths.predictions, f"data-{cfg.mln.train_on}-{seed:05d}.pt"),
                weights_only=False,
                map_location="cpu",
            )

            data_train = {
                prep(k): v for k, v in data_train.items() if k.endswith("-logits")
            }

            if filter_att:
                data_train = {
                    k: v for k, v in data_train.items() if k.split("-")[0] in filter_att
                }

            if to_half:
                data_train = {k: v.half() for k, v in data_train.items()}

            data_in = torch.load(
                join(cfg.paths.predictions, f"data-{cfg.validate_on}-{seed:05d}.pt"),
                weights_only=False,
                map_location="cpu",
            )

            data_in = {prep(k): v for k, v in data_in.items() if k.endswith("-logits")}

            if filter_att:
                # this sort of splitting might lead to errors
                data_in = {
                    k: v for k, v in data_in.items() if k.split("-")[0] in filter_att
                }

            if to_half:
                data_in = {k: v.half() for k, v in data_in.items()}

            data_ood = torch.load(
                join(cfg.paths.predictions, f"data-{cfg.dataset}-{seed:05d}.pt"),
                weights_only=False,
                map_location="cpu",
            )

            data_ood = {
                prep(k): v for k, v in data_ood.items() if k.endswith("-logits")
            }

            if filter_att:
                data_ood = {
                    k: v for k, v in data_ood.items() if k.split("-")[0] in filter_att
                }

            if to_half:
                data_ood = {k.lower(): v for k, v in data_ood.items()}
        except Exception as e:
            log.error(
                f"Could not load data: probably full memory at {socket.gethostname()}"
            )
            raise ValueError(f"Error while loading data: {e}")

        return data_in, data_ood, data_train


class FitnessAUROCOnlyPositive(FitnessAUROC):
    """
    Additionally optimize for positive constraint weights

    """

    def _compute_fitness(self, individual, metrics):
        w = self.mln.w.data
        negativeness = (w < 0).float().mean().item()
        performance = sum(metrics["AUROC"]) / len(metrics["AUROC"])
        # tokens = [len(self.compiler.tokenize(str(c))) for c in individual.constraints]
        complexity = sum(
            [c.size() for c in individual.constraints]
        )  # len(individual.constraints)
        fitness = performance - self.alpha * complexity - negativeness
        return fitness


class LogicFitness(FitnessAUROC):

    def create_detector(self, cfg, constraints, data_train):
        from detectors import LogicEnsembleDetector

        detector = LogicEnsembleDetector(
            constraints=constraints,
            ensemble_attributes=self.cfg.attributes,
            mln_attributes=self.cfg.attributes,
        )

        return detector

    def _compute_fitness(self, individual, metrics):

        performance = sum(metrics["AUROC"]) / len(metrics["AUROC"])

        complexity = sum([c.size() for c in individual.constraints])
        fitness = performance - self.alpha * complexity
        return fitness


def random_expr(propositions, operators, max_depth=2, p_unary=0.2, p_prop=0.3):
    """
    Recursively build a random logical expression from the grammar.
    - If max_depth == 0, pick a proposition or its negation.
    - Else pick a binary operator and recurse.
    """
    # Base case: random proposition or NOT proposition
    if max_depth == 0 or random.random() < p_prop:

        if isinstance(propositions, dict):
            # propositions are categorical
            prop = random.choice(list(propositions.keys()))
            value = random.choice(propositions[prop])
            atom = CategoricalVariable(name=prop, value=value)

            if random.random() < p_unary:
                return Not(atom)
            else:
                return atom

        # assume propositions are binary
        prop = random.choice(propositions)
        # chance to wrap it in NOT (or set p_unary as needed)
        if random.random() < p_unary:
            return Not(BinaryVariable(name=prop))
        else:
            return BinaryVariable(name=prop)
    else:
        # Binary operator

        left = random_expr(propositions, operators, max_depth - 1, p_unary=p_unary)
        right = random_expr(propositions, operators, max_depth - 1, p_unary=p_unary)
        op = random.choice(operators)(left=left, right=right)
        return op


def generate_random_constraint(propositions, operators, max_depth=2) -> Node:
    """
    Return a *string* that represents a constraint in your grammar.
    In practice, you would do something like:
      expr = random_expr_ast(propositions, max_depth)
      constraint_str = expr_to_string(expr)
    Here, we'll return a simple placeholder.
    """
    # stochastic depth -> will be stochastic anyway
    # depth = random.choice(range(0, max_depth))

    return random_expr(propositions, operators=operators, max_depth=max_depth)


class TreeMutation(MutationOperator):
    """
    Mutates an individual by possibly removing, adding, or replacing
    one constraint (string). Only triggers with some probability (mutation_rate).

    # TODO: implement: remove subtree
    """

    def __init__(self, propositions, operators, max_depth=2, mutation_rate=0.3):
        self.propositions = propositions
        self.max_depth = max_depth
        self.mutation_rate = mutation_rate
        self.operators = operators

    def _find_and_replace_op(
        self, current: Node, old_op: BinaryOperator, new_op: BinaryOperator
    ) -> bool:
        """
        return True: new_op is the new root
        return False: current is still the root
        """

        if current.is_leaf():
            return False

        if isinstance(current, Operator) and current.is_same(old_op):
            new_op.left = current.left
            new_op.right = current.right
            return True

        if isinstance(current, BinaryOperator):
            if self._find_and_replace_op(current.left, old_op, new_op):
                current.left = new_op.left

            if self._find_and_replace_op(current.right, old_op, new_op):
                current.right = new_op.right

            return False

        if isinstance(current, UnaryOperator):
            if self._find_and_replace_op(current.child, old_op, new_op):
                current.child = new_op

            return False

        raise ValueError()

    def _get_all_ops(self, node) -> List[BinaryOperator]:
        for child in node:
            if isinstance(child, BinaryOperator):
                yield child

    def _drop_random_subtree_rec(self, current, to_drop):
        if isinstance(current, BinaryOperator):
            if current.left == to_drop:
                if to_drop not in current._children:
                    log.warning(f"Missing child in {current}")
                    return

                prop = random_expr(self.propositions, operators=[], max_depth=0)
                # current._children.remove(to_drop)
                current.left = prop
                # current._children.append(prop)
                return

            if current.right == to_drop:
                if to_drop not in current._children:
                    log.warning(f"Missing child in {current}")
                    return

                prop = random_expr(self.propositions, operators=[], max_depth=0)
                # current._children.remove(to_drop)
                current.right = prop
                # current._children.append(prop)
                return

            self._drop_random_subtree_rec(current.left, to_drop)
            self._drop_random_subtree_rec(current.right, to_drop)

        return

    def _drop_random_subtree(self, node):
        ops = list(n for n in self._get_all_ops(node) if n != node)

        if len(ops) == 0:
            return node

        to_drop = random.choice(ops)
        self._drop_random_subtree_rec(node, to_drop)

        return node

    def mutate(self, individual: Individual) -> Individual:
        new_ind = individual.copy()
        if random.random() < self.mutation_rate:
            options = ["add", "replace", "mutate-op", "drop-op"]

            if len(individual.constraints) > 1:
                options.append("remove")

            choice = random.choice(options)

            if choice == "drop-op" and len(new_ind) > 0:
                return self.drop_op_subtree(new_ind)

            if choice == "mutate-op" and len(new_ind) > 0:
                return self.change_op(individual, new_ind)

            elif choice == "remove" and len(new_ind) > 1:
                idx = random.randrange(len(new_ind))
                del new_ind.constraints[idx]
                return new_ind

            elif choice == "add":
                new_constraint = generate_random_constraint(
                    self.propositions, self.operators, max_depth=self.max_depth
                )
                if not new_ind.has_constraint(new_constraint):
                    new_ind.add_constraint(new_constraint)

                return new_ind

            elif choice == "replace" and len(new_ind) > 0:
                return self.replace_constraint(new_ind)

        return new_ind

    def replace_constraint(self, new_ind):
        idx = random.randrange(len(new_ind))
        new_constraint = generate_random_constraint(
            self.propositions, self.operators, max_depth=self.max_depth
        )
        del new_ind.constraints[idx]
        if not new_ind.has_constraint(new_constraint):
            new_ind.add_constraint(new_constraint)

        return new_ind

    def change_op(self, individual, new_ind):
        # we change one operator in one constraint
        idx = random.randrange(len(new_ind))
        constraint = new_ind.constraints.pop(idx)
        replaced = False
        ops = self.operators[:]
        random.shuffle(ops)
        for old_op_class in ops:
            old_op = old_op_class(left=None, right=None)
            if old_op in constraint:
                other_ops = set(ops) - {old_op_class}
                new_op_class = random.choice(list(other_ops))
                new_op = new_op_class(left=None, right=None)
                constraint_copy = copy.deepcopy(constraint)

                new_op_is_root = self._find_and_replace_op(constraint, old_op, new_op)
                if new_op_is_root:
                    constraint = new_op

                log.debug(f"[{constraint_copy}] mutated to [{constraint}]")
                replaced = True
                new_ind.add_constraint(constraint)
                break
        if not replaced:
            # TODO: handle
            # log.error(
            #     f"No operator found: {','.join([str(op) for op in ops])} | {constraint}"
            # )
            new_ind = individual.copy()

        return new_ind

    def drop_op_subtree(self, new_ind):
        idx = random.randrange(len(new_ind))
        constraint = new_ind.constraints.pop(idx)
        new = self._drop_random_subtree(constraint)
        log.debug(f"drop-op constraint {constraint} to {new}")
        new_ind.add_constraint(new)
        return new_ind


def get_operators(names: List[str]) -> List:
    mapping = {
        "xor": Xor,
        "and": And,
        "or": Or,
        "implies": Implies,
    }
    return [mapping[s] for s in names]


def worker_node(cfg, objective, crossover_op, mutation_op):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    log.info(f"Hello from worker {rank}")

    log.debug(cfg.paths.predictions)

    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == 0:
            log.debug(f"Worker {rank} received task {task}")

            results = objective.fitness(task)

            log.debug(f"worker {rank} computed {results} for {task}, sending")
            # Send result back to master
            comm.send(
                {
                    "worker_id": rank,
                    "output": {
                        "individual": task,
                        "fitness": results["fitness"],
                        "result": results,
                    },
                },
                dest=0,
                tag=0,
            )

        if status.Get_tag() == 1:
            # Tag=1 signals "stop"
            log.info(f"Worker received tag, exiting.")
            break

        if status.Get_tag() == 2:
            # Tag signals crossover
            log.debug(f"Worker {rank} received task {task}")
            parent1 = task["parent1"]
            parent2 = task["parent2"]
            child = parent1

            try:
                child = crossover_op.crossover(parent1, parent2)
                # log.debug(f"crossover: {parent1} + {parent2} -> {child}")
                child = mutation_op.mutate(child)
                # log.debug(f"mutation: {child} -> {child}")
                log.debug(f"Sending result to master")
            except Exception as e:
                log.exception(e)

            comm.send(
                {
                    "worker_id": rank,
                    "output": {
                        "task": task,
                        "result": child,
                    },
                },
                dest=0,
                tag=2,
            )

    log.info(f"Worker {rank} exiting")


def mpi_create_new_population(cfg, survivors, population_size):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # total processes
    rank = comm.Get_rank()  # this should be 0 for master
    assert rank == 0, "This function should only run on the master process."

    t_start = time.time()

    new_population = copy.deepcopy(survivors)
    tasks_to_submit = population_size - len(new_population)
    individuals_to_generate = population_size - len(new_population)
    tasks_completed = 0

    log.info(
        f"Sampling new population of size {population_size} with {size - 1} MPI workers."
    )

    # 1) Pre-fill each worker with one task (if any remain)
    for worker_rank in range(1, size):
        if tasks_to_submit > 0:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            task = {"parent1": parent1, "parent2": parent2}
            comm.send(task, dest=worker_rank, tag=2)
            tasks_to_submit -= 1
        else:
            # No more tasks to send; break early
            break

    # 2) While we haven't received all children:
    while tasks_completed < individuals_to_generate:
        status = MPI.Status()
        # Receive results from any worker
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
        worker_rank = status.Get_source()

        # Process the result (the mutated child)
        child_mutated = result["output"]["result"]
        new_population.append(copy.deepcopy(child_mutated))
        tasks_completed += 1

        if tasks_completed % cfg.log_every == 0:
            log.info(
                f"Sampled {tasks_completed} in {time.time() - t_start:.2f} seconds"
            )

        # If more tasks remain, send the next one to this now-free worker
        if tasks_to_submit > 0:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            task = {"parent1": parent1, "parent2": parent2}
            comm.send(task, dest=worker_rank, tag=2)
            tasks_to_submit -= 1

    log.info(
        f"Sampling {tasks_completed} instances took {time.time() - t_start:.2f} seconds."
    )
    # The new population is now ready
    return new_population


def run_evolutionary_search(
    cfg, propositions: List[str], operators, selector, resume_from=None
):
    """
    Simple evolutionary search with parallel fitness evaluation and caching.
    Tracks population stats in a pandas DataFrame.
    """
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()

    # Global fitness cache
    fitness_cache: Dict[str, float] = {}

    if resume_from:
        checkpoint = pickle.load(open(resume_from, "rb"))
        if "fitness_cache" in checkpoint:
            fitness_cache = checkpoint["fitness_cache"]

        best_individual = checkpoint["best_individual"]
        best_fitness = checkpoint["best_fitness"]
        population = checkpoint["population"]
        generation = checkpoint["generation"] + 1
    else:
        population = init_population(
            cfg,
            cfg.init_num_constraints,
            cfg.population_size,
            propositions,
            operators,
        )

        best_individual = None
        best_fitness = -math.inf
        generation = 0

    log.info(f"Starting search")

    for generation in range(generation, cfg.n_generations):
        gen_start = time.time()
        stats_records = []

        log.info(f"=== Generation {generation} ===")
        log.info(f"=== Population {len(population)} ===")

        best_fitness, best_individual, results = mpi_run_generation(
            cfg,
            best_fitness,
            best_individual,
            comm,
            fitness_cache,
            gen_start,
            generation,
            num_workers,
            population,
        )

        population = [r["individual"] for r in results]
        fitnesses = [r["fitness"] for r in results]
        worker_outs = [r["result"] for r in results]

        # Record stats + update best
        for i, (fit, res) in enumerate(zip(fitnesses, worker_outs)):
            entry = {
                "generation": generation,
                "individual_idx": i,
                "fitness": fit,
                "size": len(population[i]),
                "signature": population[i].signature(),
            }
            entry.update(res)
            stats_records.append(entry)

        # 3) Selection (top half, for example)
        with Timer("Selecting Survivors"):
            survivors = selector.select(population, worker_outs)

        # 4) Generate new offspring via crossover + mutation
        with Timer("Sampling new population"):
            log.info(
                f"Sampling new population: {cfg.population_size} with {mp.cpu_count()}"
            )
            new_population = mpi_create_new_population(
                cfg, survivors, cfg.population_size
            )
            population = new_population

        # Save stats
        with Timer("Saving stats"):
            path = join(get_output_dir(), f"stats-{generation:05d}.csv")
            log.info(f"Saving stats to {path}")
            pd.DataFrame(stats_records).to_csv(path, index=False)

        # Save state
        with Timer("Saving state"):
            path = join(get_output_dir(), f"state-{generation:05d}.pkl")
            log.info(f"Saving state to {path}")

            pickle.dump(
                {
                    # "cache": fitness_cache, # do not save cache because it blows up memory
                    "population": population,
                    "best_individual": best_individual,
                    "best_fitness": best_fitness,
                    "generation": generation,
                },
                open(path, "wb"),
            )

    # log.info(
    #     f"Finished. Best overall fitness: {best_fitness:.4f} with {len(best_individual.constraints)} constraints."
    # )
    return best_individual, best_fitness


def init_population(
    cfg,
    init_constraints_per_individual,
    population_size,
    propositions,
    operators,
):
    population: List[Individual] = []
    log.info(f"Creating population")
    for n in range(population_size):
        # log.info(f"Creating {n}")
        constraints = []
        for j in range(init_constraints_per_individual):
            c = generate_random_constraint(
                propositions, operators, max_depth=cfg.max_depth
            )
            # log.info(f"Generation {n}: {c}")
            constraints.append(c)

        # log.info(f"Pop {n}/{population_size}")
        population.append(Individual(constraints))
    return population


def mpi_run_generation(
    cfg,
    best_fitness,
    best_individual,
    comm,
    fitness_cache,
    gen_start,
    generation,
    num_workers,
    population,
):
    # Evaluate fitness in parallel
    num_tasks = len(population)
    active_workers = 0
    next_task_index = 0
    results = []
    cache_hit_counter = 0
    error_counter = 0

    # best_performance = 0  # TODO

    # Send 1 initial task to each worker (if enough tasks)
    for worker_id in range(1, num_workers):
        while next_task_index < num_tasks:
            r = fitness_cache.get(population[next_task_index].signature())
            if r:
                results.append(r)
                next_task_index += 1
                cache_hit_counter += 1
            else:
                # log.debug(
                #     f"Submitting {population[next_task_index]} to worker{worker_id}"
                # )
                comm.send(population[next_task_index], dest=worker_id, tag=0)
                next_task_index += 1
                active_workers += 1
                break
        # else:
        #     active_workers -= 1

    # Collect results and keep distributing until tasks are done
    while active_workers > 0:
        log.debug(f"Waiting for remaining {active_workers} workers")
        # Receive result from a worker
        result = comm.recv(source=MPI.ANY_SOURCE, tag=0)
        worker_id = result["worker_id"]
        active_workers -= 1

        output = result["output"]
        if output["fitness"] == 0:
            error_counter += 1

        # log.debug(f"Received {output['individual']} from worker{worker_id}")

        fitness_cache[output["individual"].signature()] = output

        if output["fitness"] > best_fitness:
            best_fitness = output["fitness"]
            # best_individual = output["individual"].copy()
            # best_performance = output["result"]["performance"]

        results.append(output)

        if len(results) % cfg.log_every == 0:
            try:
                log.info(
                    f"Epoch {generation:03d} Finished {len(results):05d}/{len(population):05d} {len(results) / len(population):.2%} in {time.time() - gen_start:.1f}s [Fitness: {best_fitness:.5f}] cached: {len(fitness_cache)} hit: {cache_hit_counter} err: {error_counter}"
                )
            except Exception as e:
                log.warning("Formatting error")

        # If there are still tasks left, send next task
        while next_task_index < num_tasks:
            r = fitness_cache.get(population[next_task_index].signature())
            if r:
                results.append(r)
                next_task_index += 1
                cache_hit_counter += 1
            else:
                # log.debug(
                #     f"Submitting {population[next_task_index]} to worker{worker_id}"
                # )
                comm.send(population[next_task_index], dest=worker_id, tag=0)
                next_task_index += 1
                active_workers += 1
                break
        # else:
        #     active_workers -= 1
    return best_fitness, best_individual, results
