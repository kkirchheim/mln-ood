import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List

import hydra
import pandas as pd
from omegaconf import OmegaConf

from detectors import *
from shared import train_mln
from mln import MLN
from dataset import GTSRB

from utils import get_constraints, wrap_shared_model

log = logging.getLogger(__name__)


class LLMDetector(Detector):
    def __init__(self, llm_results_files, name):
        super().__init__(name)
        self.llm_results_files = llm_results_files
        self.df = pd.read_csv(self.llm_results_files)
        labels = list(GTSRB.class_to_name.values())
        shapes = list(GTSRB.shape_to_name.values())
        colors = list(GTSRB.color_to_name.values())

        self.df["label"] = self.df["label"].apply(labels.index)
        self.df["color"] = self.df["color"].apply(colors.index)
        self.df["shape"] = (
            self.df["shape"].apply(lambda x: x.replace("_", " ")).apply(shapes.index)
        )

        self.df = self.df.set_index(["label", "color", "shape"])

    def get_llm_decision(self, label, color, shape):
        return int(self.df.loc[label, color, shape]["decision"])

    def predict(self, cfg, data):
        ensemble_scores = 1 - (
            torch.stack(
                [
                    data["label-logits"].softmax(dim=1).max(dim=1).values,
                    data["color-logits"].softmax(dim=1).max(dim=1).values,
                    data["shape-logits"].softmax(dim=1).max(dim=1).values,
                ],
                dim=1,
            )
            .mean(dim=1, keepdim=True)
            .squeeze()
        )

        labels = torch.stack(
            [
                data["label-logits"].max(dim=1).indices,
                data["color-logits"].max(dim=1).indices,
                data["shape-logits"].max(dim=1).indices,
            ],
            dim=1,
        )

        # Convert the tensor to a pandas DataFrame for batch lookup
        labels_df = pd.DataFrame(labels.numpy(), columns=["label", "color", "shape"])

        mydataframe_reset = self.df.reset_index()
        merged_df = pd.merge(
            labels_df, mydataframe_reset, on=["label", "color", "shape"], how="left"
        )

        scores = merged_df["decision"].to_numpy()

        scores = torch.tensor(scores, dtype=torch.bool)
        scores = torch.where(scores, ensemble_scores.max().item(), ensemble_scores)
        return scores


def constraint_is_id(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x[:, 3] == 1, 1, 0).unsqueeze(1)


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


def fit_all_detectors(cfg, seed) -> List[Detector]:
    constraints = get_constraints(cfg)[:]
    detectors = []

    data = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.mln.train_on}-{seed:05d}.pt"),
        weights_only=False,
    )

    oe_mln = None
    if cfg.use_oe:
        data_oe = torch.stack(
            [
                data["label-logits"].max(dim=1).indices,
                data["color-logits"].max(dim=1).indices,
                data["shape-logits"].max(dim=1).indices,
                data["oe-logits"].max(dim=1).indices,
            ],
            dim=1,
        )

        oe_mln = MLN(
            constraints=constraints + [constraint_is_id],
            domain=[range(43), [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1]],
        )
        train_mln(cfg, oe_mln, data_oe, use_prog=cfg.mln.progress)

        torch.save(oe_mln.w.data, join(get_output_dir(), f"oe-mln-{seed}.pt"))

    data_normal = torch.stack(
        [
            data["label-logits"].max(dim=1).indices,
            data["color-logits"].max(dim=1).indices,
            data["shape-logits"].max(dim=1).indices,
        ],
        dim=1,
    )

    if "ablation_n_rules" in cfg and cfg.ablation_n_rules:
        log.info(f"Limiting number of rules")
        constraints = constraints[: cfg.ablation_n_rules]

    mln = MLN(
        constraints=constraints, domain=[range(43), [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    )
    train_mln(cfg, mln, data_normal, use_prog=cfg.mln.progress)
    torch.save(mln.w.data, join(get_output_dir(), f"mln-weights-{seed}.pt"))

    detectors.extend(initialize_detectors(cfg, mln, seed, oe_mln))

    return detectors


def initialize_detectors(cfg, mln, seed, oe_mln=None) -> List[Detector]:
    # Load model to get w, b (weights and biases) for ViM, DICE
    if cfg.is_shared:
        # special preparations for model with shared encoder
        label_net = torch.load(
            join(cfg.paths.models, f"model-shared-{seed:05d}.pt"), weights_only=False
        )
        label_net = wrap_shared_model(label_net)
    else:
        label_net = torch.load(
            join(cfg.paths.models, f"model-label-{seed:05d}.pt"), weights_only=False
        )

    w, b = get_weights_and_biases(label_net)

    # these have to be fitted first and can then be reused
    vim = ViMDetector(w, b)
    dice = DICEDetector(w, b)
    maha = MahalanobisDetector()

    head = torch.nn.Linear(w.shape[0], w.shape[1], dtype=torch.float32)
    head.weight.data = w.float()
    head.bias.data = b.float()
    head = head.to(cfg.device)

    react = ReActDetector(head=head)
    she = SHEDetector(head=head)

    attributes = ["label", "color", "shape"]
    if cfg.use_oe:
        attributes.append("oe")

    detectors: List[Detector] = [
        vim,
        dice,
        maha,
        react,
        she,
        MSPDetector(),
        EntropyDetector(),
        EBODetector(),
        EnsembleDetector(),
        LogicDetector(constraints=mln.constraints, attributes=attributes),
        MLNDetector(mln, attributes=attributes),
        # MLNUntrainedDetector(constraints=mln.constraints, name="MLN-Untrained", attributes=attributes),
        LogicEnsembleDetector(constraints=mln.constraints, mln_attributes=attributes),
        MLNEnsembleDetector(mln, mln_attributes=attributes, dist=cfg.fit_score_dist),
        MLNViMDetector(mln, vim, mln_attributes=attributes, dist=cfg.fit_score_dist),
        MLNEBODetector(mln, mln_attributes=attributes, dist=cfg.fit_score_dist),
        MLNDICEDetector(mln, dice, mln_attributes=attributes, dist=cfg.fit_score_dist),
        MLNMahalanobisDetector(
            mln, maha, mln_attributes=attributes, dist=cfg.fit_score_dist
        ),
        MLNMSPDetector(mln, mln_attributes=attributes, dist=cfg.fit_score_dist),
        MLNReActDetector(
            mln, react, mln_attributes=attributes, dist=cfg.fit_score_dist
        ),
        MLNSHEDetector(mln, she, mln_attributes=attributes, dist=cfg.fit_score_dist),
    ]

    if oe_mln:
        detectors += [
            MLNDetector(oe_mln, name="MLN+"),
            LogicDetector(constraints=oe_mln.constraints, name="Logic+"),
            LogicEnsembleDetector(
                constraints=oe_mln.constraints, name="Logic+Ensemble+"
            ),
            MLNEnsembleDetector(oe_mln, name="MLN+Ensemble+", dist=cfg.fit_score_dist),
            MLNViMDetector(oe_mln, vim, name="MLN+ViM+", dist=cfg.fit_score_dist),
            MLNEBODetector(oe_mln, name="MLN+EBO+", dist=cfg.fit_score_dist),
            MLNDICEDetector(oe_mln, dice, name="MLN+DICE+", dist=cfg.fit_score_dist),
            MLNMahalanobisDetector(
                oe_mln, maha, name="MLN+Mahalanobis+", dist=cfg.fit_score_dist
            ),
            MLNMSPDetector(oe_mln, name="MLN+MSP+"),
            MLNReActDetector(
                oe_mln,
                react,
                name="MLN+ReAct+",
                mln_attributes=attributes,
                dist=cfg.fit_score_dist,
            ),
            MLNSHEDetector(
                oe_mln,
                she,
                name="MLN+SHE+",
                mln_attributes=attributes,
                dist=cfg.fit_score_dist,
            ),
        ]

    if cfg.use_llm:
        for key, value in cfg["llm_results_files"].items():
            detectors += [LLMDetector(name=key, llm_results_files=value)]

    fitting_data = torch.load(
        join(cfg.paths.predictions, f"data-{cfg.fit_score_dist_on}-{seed:05d}.pt"),
        weights_only=False,
    )
    for detector in detectors:
        detector.fit(cfg, fitting_data)

    return detectors


def evaluate_all(cfg, seed, detectors: List[Detector]) -> pd.DataFrame:

    id_data = torch.load(
        join(cfg.paths.predictions, f"data-test-{seed:05d}.pt"), weights_only=False
    )

    # id_data = {prep(k): v for k, v in id_data.items()}

    result_dfs = []

    for dataset_name in cfg.datasets:
        log.info(f"Evaluating on dataset '{dataset_name}'")
        ood_data = torch.load(
            join(cfg.paths.predictions, f"data-{dataset_name}-{seed:05d}.pt"),
            weights_only=False,
        )

        # ood_data = {prep(k): v for k, v in ood_data.items()}

        for detector in detectors:
            metrics = detector.evaluate(
                cfg, id_data, ood_data, dataset_name=dataset_name
            )
            result = pd.DataFrame([{k: v * 100 for k, v in metrics.compute().items()}])
            result["Dataset"] = dataset_name
            result["Method"] = detector.name
            result["Seed"] = seed
            result_dfs.append(result)

    return pd.concat(result_dfs)


if __name__ == "__main__":
    main()
