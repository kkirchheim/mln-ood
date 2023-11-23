import logging
import os
import time
import sys
from os.path import dirname, join
import torch

sys.path.append(join(dirname(__file__), ".."))

import hydra

try:
    from mpi4py import MPI
except ImportError:
    print(f"mpi not installed, there might be problems")

from compiler import ConstraintCompiler
from evo import (
    RandomSubsetCrossover,
    TreeMutation,
    get_operators,
    worker_node,
    run_evolutionary_search,
)
from shared import get_output_dir, get_machine_lock_file
from pytorch_ood.utils import fix_random_seed


log = logging.getLogger(f"rank-{MPI.COMM_WORLD.Get_rank():03d}")


@hydra.main(config_path="config", config_name="evo-mining.yaml", version_base="1.2")
def main(cfg):
    fix_random_seed(cfg.seed)
    torch.multiprocessing.set_start_method("spawn", force=True)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Safety check (need at least 2 ranks: 1 master + 1 worker)
    if size < 2:
        if rank == 0:
            log.error("Error: Run with at least 2 processes.")
        return

    attribute_index_map = {
        k.replace("class", "label"): v for v, k in enumerate(cfg.attributes)
    }
    compiler = ConstraintCompiler(attribute_index_map)

    propositions = cfg.attributes
    operators = get_operators(cfg.operators)

    mutation_op = TreeMutation(
        propositions=propositions,
        operators=operators,
        max_depth=cfg.max_depth,
        mutation_rate=cfg.mutation_rate,
    )

    selector = hydra.utils.instantiate(cfg.selector)

    crossover_op = RandomSubsetCrossover()

    domain = [[0, 1] for _ in range(len(attribute_index_map.keys()))]

    objective = hydra.utils.instantiate(cfg.objective, domain=domain, compiler=compiler)
    objective.cfg = cfg

    if rank == 0:
        if "debug" in cfg and cfg.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)

        master_node(cfg, propositions, operators, selector=selector)
        time.sleep(5)
        log.info(f"Terminating master")
    else:
        try:
            if "debug" in cfg and cfg.debug:
                logging.getLogger().setLevel(logging.DEBUG)
                log.setLevel(logging.DEBUG)

            worker_node(
                cfg,
                objective=objective,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
            )
        except Exception as e:
            log.exception(e)
            raise e

    # clear old file locks
    if os.path.exists(get_machine_lock_file()):
        os.remove(get_machine_lock_file())


def master_node(cfg, propositions, operators, selector):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    active_workers = size - 1
    log.info(f"Hello from master, workers: {active_workers}")
    log.info(f"{get_output_dir()}")

    time.sleep(4)

    if "resume_from" in cfg:
        resume_from = cfg.resume_from
    else:
        resume_from = None

    _ = run_evolutionary_search(
        cfg=cfg,
        propositions=propositions,
        operators=operators,
        selector=selector,
        resume_from=resume_from,
    )

    for worker_id in range(1, size):
        log.info(f"Terminating worker {worker_id}")
        comm.send(None, dest=worker_id, tag=1)


if __name__ == "__main__":
    main()
