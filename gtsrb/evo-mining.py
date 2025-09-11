import logging
import time

import hydra

try:
    from mpi4py import MPI
except ImportError:
    print(f"mpi not installed, there might be problems")
import multiprocessing as mp

from compiler import ConstraintCompiler
from evo import (
    RandomSubsetCrossover,
    TreeMutation,
    get_operators,
    worker_node,
    run_evolutionary_search,
    RandomSampler,
    LLMSampler
)
from shared import get_output_dir
from dataset import category_value_map


log = logging.getLogger(f"rank-{MPI.COMM_WORLD.Get_rank():03d}")

logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# hotfix
category_value_map = {
    k.replace("class", "label"): v for k, v in category_value_map.items()
}


@hydra.main(config_path="config", config_name="evo-mining.yaml", version_base="1.2")
def main(cfg):
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
    compiler = ConstraintCompiler(
        attribute_index_map, category_value_map=category_value_map
    )

    propositions = {k: list(v.keys()) for k, v in category_value_map.items()}
    operators = get_operators(cfg.operators)

    mutation_op = TreeMutation(
        propositions=propositions,
        operators=operators,
        max_depth=cfg.max_depth,
        mutation_rate=cfg.mutation_rate,
    )

    selector = hydra.utils.instantiate(cfg.selector)

    crossover_op = RandomSubsetCrossover()

    domain = [range(43), [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1]]

    objective = hydra.utils.instantiate(cfg.objective, domain=domain, compiler=compiler)
    objective.cfg = cfg

    sampler = LLMSampler(
        propositions=propositions,
        operators=operators,
        compiler=compiler,
    )

    if rank == 0:
        if "debug" in cfg and cfg.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)

        master_node(cfg, propositions, operators, selector=selector, sampler=sampler)
        time.sleep(5)
        log.info(f"Terminating master")
    else:
        if "debug" in cfg and cfg.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            log.setLevel(logging.DEBUG)

        worker_node(
            cfg, objective=objective, crossover_op=crossover_op, mutation_op=mutation_op
        )


def master_node(cfg, propositions, operators, selector, sampler):
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
        sampler=sampler,
        resume_from=resume_from,
    )

    for worker_id in range(1, size):
        log.info(f"Terminating worker {worker_id}")
        comm.send(None, dest=worker_id, tag=1)


if __name__ == "__main__":
    # Set start method to 'spawn' to avoid memory copy issue
    mp.set_start_method("spawn", force=True)
    main()
