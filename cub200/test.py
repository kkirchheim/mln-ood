import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler import ConstraintCompiler

from typing import List

import hydra
import pandas as pd
from omegaconf import OmegaConf


from shared import train_mln, train_mln_pseudo
from detectors import *
from mln import MLN

log = logging.getLogger(__name__)


def get_domain(attributes):
    domain = []
    for att in attributes:
        if att == "class":
            domain.append(range(200))
        else:
            domain.append(range(2))

    return domain


@hydra.main(config_path="config", config_name="test.yaml", version_base="1.2")
def main(cfg):
    log.info("\n" + OmegaConf.to_yaml(cfg))
    log.info(f"Output-dir: {get_output_dir()}")
    try:
        results = []
        for seed in range(cfg.n_seeds):
            log.info(f"Seed: {seed}")
            detectors = fit_all_detectors(cfg, seed)
            result = evaluate_all(cfg, seed, detectors)
            results.append(result)

        df = pd.concat(results)
        df.to_pickle(join(get_output_dir(), "results.pkl"))
        df.to_csv(join(get_output_dir(), "results.csv"))

        agg_results = (
            df.groupby(["Method", "Dataset"])
            .agg(["mean"])
            .groupby(["Method"])
            .agg(["mean", "sem"])[["AUROC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]]
        )
        log.info("\n" + agg_results.to_string())

        agg_results.to_pickle(join(get_output_dir(), "results-agg.pkl"))
        agg_results.to_csv(join(get_output_dir(), "results-agg.csv"))

        r = agg_results.loc["MLN+Ensemble"]["AUROC"]["mean"]["mean"]

        # writer = SummaryWriter(get_output_dir())
        # writer.add_hparams(dict(cfg.mln), metric_dict={"AUROC": r})

        return r

    except Exception as e:
        log.exception(e)
    return float("-inf")


def get_constraints(cfg: OmegaConf):
    attribute_index_map = {att: n for n, att in enumerate(cfg.attributes)}
    log.info(attribute_index_map)
    compiler = ConstraintCompiler(attribute_index_map)

    # Create a dedicated namespace dictionary
    namespace = {}
    global_namespace = {
        "torch": torch,
    }

    for constraint_str in cfg.constraints:
        constraint_name = (
            constraint_str.replace(" ", "_")
            .replace("->", "implies")
            .replace("=", "_eq_")
            .replace("(", "_")
            .replace(")", "_")
        )

        generated_code = compiler.compile(constraint_name, constraint_str)
        log.info(f"CODE:\n{generated_code}")
        namespace[constraint_name] = generated_code
        # Execute the generated code within this namespace
        exec(generated_code, global_namespace, namespace)

    rule_fn = list(namespace.values())
    return rule_fn


def prep_data(data):
    data = {
        key.lower()
        .replace("-", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("_logits", "-logits")
        .replace("_features", "-features")
        .replace("_labels", "-labels"): value
        for key, value in data.items()
    }
    data["labels"] = data["class-labels"]
    return data


def fit_all_detectors(cfg, seed) -> List[Detector]:
    constraints = get_constraints(cfg)[:]
    detectors = []

    data = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.mln.train_on}-{seed:05d}.pt"),
        weights_only=False,
    )
    data = prep_data(data)

    if "ablation_n_rules" in cfg and cfg.ablation_n_rules:
        log.info(f"Limiting number of rules")
        constraints = constraints[: cfg.ablation_n_rules]

    oe_mln = None
    if cfg.use_oe:
        data_oe = get_predictions(data, cfg.attributes)

        def constraint_is_id(x: torch.Tensor) -> torch.Tensor:
            return torch.where(x[:, (len(cfg.attributes) - 1)] == 1, 1, 0).unsqueeze(1)

        oe_mln = MLN(
            constraints=constraints + [constraint_is_id],
            domain=get_domain(cfg.attributes),
        )
        train_mln(cfg, oe_mln, data_oe)
        torch.save(oe_mln.w.data, join(get_output_dir(), f"oe-mln-{seed}.pt"))

    log.info(list(data.keys()))

    attributes = [a for a in cfg.attributes if a != "oe"]
    log.info(attributes)
    data_normal = get_predictions(data, attributes)
    mln = MLN(constraints=constraints, domain=get_domain(attributes))

    if cfg.mln.loss == "likelihood":
        train_mln(cfg, mln, data_normal)
    elif cfg.mln.loss == "pseudo":
        train_mln_pseudo(cfg, mln, data_normal)

    torch.save(mln.w.data, join(get_output_dir(), f"mln-{seed}.pt"))

    detectors.extend(initialize_detectors(cfg, mln, seed, oe_mln))

    return detectors


def initialize_detectors(cfg, mln, seed, oe_mln=None) -> List[Detector]:

    label_net = torch.load(
        join(cfg.paths.models, f"model-{cfg.target_att}-{seed:05d}.pt"),
        weights_only=False,
    )

    w, b = get_weights_and_biases(label_net)

    # these have to be fitted first and can then be reused
    vim = ViMDetector(w, b, attribute=cfg.target_att)
    dice = DICEDetector(w, b, attribute=cfg.target_att)
    maha = MahalanobisDetector(attribute=cfg.target_att)

    # omit last for oe
    ensemble_atts = cfg.attributes[:-1]
    log.info(f"Ensemble attributes: {ensemble_atts}")

    detectors: List[Detector] = [
        vim,
        dice,
        maha,
        MSPDetector(attribute=cfg.target_att),
        EntropyDetector(attribute=cfg.target_att),
        EBODetector(attribute=cfg.target_att),
        EnsembleDetector(attributes=ensemble_atts),
        LogicDetector(constraints=mln.constraints, attributes=ensemble_atts),
        MLNDetector(mln, name="MLN", attributes=ensemble_atts),
        # MLNUntrainedDetector(
        #     constraints=mln.constraints, name="MLN-Untrained", attributes=cfg.attributes, domain=mln.domain
        # ),
        LogicEnsembleDetector(
            constraints=mln.constraints,
            name="Logic+Ensemble",
            ensemble_attributes=ensemble_atts,
            mln_attributes=ensemble_atts,
        ),
        MLNEnsembleDetector(
            mln,
            ensemble_attributes=ensemble_atts,
            mln_attributes=ensemble_atts,
        ),
        MLNViMDetector(
            mln, vim, mln_attributes=ensemble_atts, attribute=cfg.target_att
        ),
        MLNEBODetector(mln, mln_attributes=ensemble_atts, attribute=cfg.target_att),
        MLNDICEDetector(
            mln, dice, mln_attributes=ensemble_atts, attribute=cfg.target_att
        ),
        MLNMahalanobisDetector(
            mln, maha, mln_attributes=ensemble_atts, attribute=cfg.target_att
        ),
        MLNMSPDetector(mln, mln_attributes=ensemble_atts, attribute=cfg.target_att),
    ]

    if oe_mln:
        detectors += [
            MLNDetector(oe_mln, name="MLN+", attributes=cfg.attributes),
            LogicDetector(
                constraints=oe_mln.constraints, name="Logic+", attributes=cfg.attributes
            ),
            LogicEnsembleDetector(
                constraints=oe_mln.constraints,
                name="Logic+Ensemble+",
                mln_attributes=cfg.attributes,
                ensemble_attributes=ensemble_atts,
            ),
            MLNEnsembleDetector(
                oe_mln,
                name="MLN+Ensemble+",
                mln_attributes=cfg.attributes,
                ensemble_attributes=ensemble_atts,
            ),
            MLNViMDetector(
                oe_mln,
                vim,
                name="MLN+ViM+",
                mln_attributes=cfg.attributes,
                attribute=cfg.target_att,
            ),
            MLNEBODetector(
                oe_mln,
                name="MLN+EBO+",
                mln_attributes=cfg.attributes,
                attribute=cfg.target_att,
            ),
            MLNDICEDetector(
                oe_mln,
                dice,
                name="MLN+DICE+",
                mln_attributes=cfg.attributes,
                attribute=cfg.target_att,
            ),
            MLNMahalanobisDetector(
                oe_mln,
                maha,
                name="MLN+Mahalanobis+",
                mln_attributes=cfg.attributes,
                attribute=cfg.target_att,
            ),
            MLNMSPDetector(
                oe_mln,
                name="MLN+Ensemble+",
                attribute=cfg.target_att,
                mln_attributes=cfg.attributes,
            ),
        ]

    fitting_data = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.fit_score_dist_on}-{seed:05d}.pt"),
        weights_only=False,
    )

    fitting_data = prep_data(fitting_data)

    for detector in detectors:
        detector.fit(cfg, fitting_data)

    return detectors


def evaluate_all(cfg, seed, detectors: List[Detector]) -> pd.DataFrame:

    id_data = torch.load(
        join(cfg.paths.predictions, f"data-test-{seed:05d}.pt"), weights_only=False
    )
    id_data = prep_data(id_data)

    result_dfs = []

    for dataset_name in cfg.datasets:
        log.info(f"Evaluating on dataset '{dataset_name}'")
        ood_data = torch.load(
            join(cfg.paths.predictions, f"data-{dataset_name}-{seed:05d}.pt"),
            weights_only=False,
        )
        ood_data = prep_data(ood_data)

        for detector in detectors:
            metrics = detector.evaluate(
                cfg, id_data, ood_data, dataset_name=dataset_name
            )

            if hasattr(detector, "to"):
                detector.to("cpu")

            result = pd.DataFrame([{k: v * 100 for k, v in metrics.compute().items()}])
            result["Dataset"] = dataset_name
            result["Method"] = detector.name
            result["Seed"] = seed
            result_dfs.append(result)

    return pd.concat(result_dfs)


if __name__ == "__main__":
    main()
