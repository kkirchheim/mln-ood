import logging
from os.path import join
from typing import Dict

import pytorch_ood.api
import torch
from pytorch_ood.detector import (
    MaxSoftmax,
    Entropy,
    EnergyBased,
    Mahalanobis,
    ViM,
    DICE,
    SHE,
    ReAct,
)
from pytorch_ood.utils import OODMetrics
from scipy.stats import genextreme, cramervonmises
from torch.utils.data import TensorDataset, DataLoader
import scipy
import numpy as np

from shared import get_output_dir

log = logging.getLogger(__name__)


class Detector:
    def __init__(self, name):
        self.name = name

    def fit(self, cfg, fitting_data):
        pass

    def predict(self, cfg, in_data_raw):
        # Should return a tensor of scores (higher = more likely OOD)
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(
        self, cfg, id_data: Dict, ood_data: Dict, dataset_name: str = None
    ) -> OODMetrics:
        """ """
        # log.debug(f"Evaluating {self.name}")
        metrics = OODMetrics()
        scores_id = self.predict(cfg, id_data).squeeze()
        scores_ood = self.predict(cfg, ood_data).squeeze()

        # sometimes, values become tiny, and pytorch-ood can not hande doubles well
        # so we normalize. this does not change the order of the score and does thus not influence metrics like auroc
        # fpr95 etc.

        scores_min = torch.cat([scores_id, scores_ood]).min().item()
        scores_max = torch.cat([scores_id, scores_ood]).max().item()

        if scores_max == scores_min:
            factor = scores_max
        else:
            factor = 1 / (scores_max - scores_min)

        scores_id *= factor
        scores_ood *= factor

        metrics.update(scores_id.float(), torch.ones(scores_id.shape[0]))
        metrics.update(scores_ood.float(), -torch.ones(scores_ood.shape[0]))

        if dataset_name:
            torch.save(
                {
                    "ID": scores_id,
                    "OOD": scores_ood,
                },
                join(
                    get_output_dir(),
                    f"scores-{dataset_name.lower()}-{self.name.lower()}.pt",
                ),
            )

        return metrics


@torch.no_grad()
def batch_predict_features(
    detector: pytorch_ood.api.Detector, features, batch_size, device
):
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    for batch in dataloader:
        batch_features = batch[0].to(device).float()
        # log.info(f"Batch size {batch_features.shape[0]}, device {batch_features.device}, precision {batch_features.dtype}")
        predictions = detector.predict_features(batch_features)
        all_predictions.append(predictions)

    return torch.cat(all_predictions, dim=0).cpu()


class MSPDetector(Detector):
    def __init__(self, attribute="label"):
        super().__init__("MSP")
        self.detector = MaxSoftmax(model=None)
        self.attribute = attribute

    def predict(self, cfg, data):
        return self.detector.predict_features(data[f"{self.attribute}-logits"])


class EntropyDetector(Detector):
    def __init__(self, attribute="label"):
        super().__init__("Entropy")
        self.detector = Entropy(model=None)
        self.attribute = attribute

    def predict(self, cfg, data):
        return self.detector.predict_features(data[f"{self.attribute}-logits"])


class EBODetector(Detector):
    def __init__(self, attribute="label"):
        super().__init__("EBO")
        self.detector = EnergyBased(model=None)
        self.attribute = attribute

    def predict(self, cfg, data):
        return self.detector.predict_features(data[f"{self.attribute}-logits"])


class MahalanobisDetector(Detector):
    def __init__(self, attribute="label", target_att_index=0):
        super().__init__("Mahalanobis")
        self.detector = Mahalanobis(model=None)
        self.attribute = attribute
        self.target_att_index = target_att_index

    def fit(self, cfg, data):
        # log.info(
        #     f"Fitting {self.name} on {self.attribute} with {data[f'{self.attribute}-features'].shape}"
        # )
        log.info(f"Mahalanobis Classes {len(data['labels'].unique().tolist())}")

        self.detector.fit_features(
            data[f"{self.attribute}-features"],
            data["labels"][:, self.target_att_index],
            device=cfg.device,
        )

    def predict(self, cfg, data):
        return batch_predict_features(
            self.detector, data[f"{self.attribute}-features"], 1024, cfg.device
        )


class ViMDetector(Detector):
    def __init__(self, w, b, attribute="label", target_att_index=0):
        super().__init__("ViM")
        self.detector = ViM(model=None, d=64, w=w, b=b)
        self.attribute = attribute
        self.target_att_index = target_att_index

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name}")
        self.detector.fit_features(
            fitting_data[f"{self.attribute}-features"],
            fitting_data["labels"][:, self.target_att_index],
        )

    def predict(self, cfg, data):
        predictions = self.detector.predict_features(
            data[f"{self.attribute}-features"].to(cfg.device)
        )
        return predictions


class DICEDetector(Detector):
    def __init__(self, w, b, attribute="label", target_att_index=0):
        super().__init__("DICE")
        self.detector = DICE(model=None, w=w, b=b, p=0.25)
        # self.params = None
        self.attribute = attribute
        self.target_att_index = target_att_index

    def fit(self, cfg, data):
        log.info(f"Fitting {self.name}")
        self.detector.fit_features(
            data[f"{self.attribute}-features"],
            data["labels"][:, self.target_att_index],
        )
        # scores_val = self.detector.predict_features(data[f"{self.attribute}-features"])
        # self.params = genextreme.fit(scores_val.cpu())

    def predict(self, cfg, data):
        predictions = batch_predict_features(
            self.detector, data[f"{self.attribute}-features"], 1024, cfg.device
        )
        return predictions


class SHEDetector(Detector):
    def __init__(self, head, attribute="label", target_att_index=0):
        super().__init__("SHE")
        self.detector = SHE(backbone=None, head=head)
        # self.params = None
        self.attribute = attribute
        self.target_att_index = target_att_index

    def fit(self, cfg, data):
        log.info(f"Fitting {self.name} on classes: {data['labels'][:, self.target_att_index].unique()} with {data['labels'].shape=}")
        self.detector.fit_features(
            data[f"{self.attribute}-features"].float(),
            data["labels"][:, self.target_att_index],
            device=cfg.device,
        )

    def predict(self, cfg, data):
        predictions = batch_predict_features(
            self.detector, data[f"{self.attribute}-features"], 1024, cfg.device
        )
        return predictions


class ReActDetector(Detector):
    def __init__(self, head, attribute="label"):
        super().__init__("ReAct")
        self.detector = ReAct(backbone=None, head=head)
        self.attribute = attribute

    def predict(self, cfg, data):
        predictions = batch_predict_features(
            self.detector, data[f"{self.attribute}-features"], 1024, cfg.device
        )
        return predictions


class MLNDetector(Detector):
    def __init__(self, mln, name="MLN", attributes=None):
        super().__init__(name)
        if attributes is None:
            attributes = ["label", "color", "shape", "oe"]
        self.attributes = attributes

        self.mln = mln

    def predict(self, cfg, data):
        predictions = get_predictions(data, self.attributes)
        scores = self.mln.energy(predictions.to(cfg.device)).cpu()
        return -scores


class EnsembleDetector(Detector):
    def __init__(self, attributes=None):
        super().__init__("Ensemble")
        if attributes is None:
            attributes = ["label", "color", "shape"]
        self.attributes = attributes

    def predict(self, cfg, in_data_raw):
        score = ensemble_score(in_data_raw, self.attributes)
        return score


class LogicDetector(Detector):
    def __init__(self, constraints, name="Logic", attributes=None):
        super().__init__(name)
        if attributes is None:
            attributes = ["label", "color", "shape", "oe"]

        self.attributes = attributes

        self.constraints = constraints

    def predict(self, cfg, data_raw):
        predictions = get_predictions(data_raw, self.attributes)
        valid = torch.stack([f(predictions) for f in self.constraints], dim=1).all(
            dim=1
        )
        return -valid.float()


def get_weights_and_biases(nn):
    if hasattr(nn, "fc"):
        return nn.fc.weight, nn.fc.bias
    elif hasattr(nn, "fc0"):
        # shared model
        return nn.fc0.weight, nn.fc0.bias
    elif hasattr(nn, "classifier"):
        return nn.classifier[-1].weight, nn.classifier[-1].bias
    elif hasattr(nn, "trafo"):
        return nn.trafo.heads[-1].weight, nn.trafo.heads[-1].bias
    else:
        raise ValueError("No known classifier head found.")


def get_predictions(data, attributes):
    predictions = torch.stack(
        [data[f"{att}-logits"].max(dim=1).indices for att in attributes],
        dim=1,
    )
    return predictions


class MLNCombinedDetector(Detector):
    """
    Base class for detectors that combine MLN probabilities with another OOD score (like ViM, DICE, etc.)
    and apply a genextreme calibration on the second OOD score.
    """

    def __init__(
        self,
        name,
        mln,
        second_detector_predict_func,
        mln_attributes=None,
        dist="genextreme",
    ):
        super().__init__(name)
        if mln_attributes is None:
            mln_attributes = ["label", "color", "shape", "oe"]

        self.mln_attributes = mln_attributes

        self.mln = mln
        self.second_predict = second_detector_predict_func
        self.params = None

        self.dist_name = dist

        if dist is None or dist == "None":

            class NoDist:
                def fit(self, x):
                    return "dummy"

                def sf(self, x, *args, **kwargs):
                    return np.ones(x.shape[0], dtype=np.float32)

            self.dist = NoDist()
        elif dist == "auto":
            # setting dist to none will trigger hpo of distribution based on goodness of fit
            self.dist = None
        else:
            self.dist = getattr(scipy.stats, dist)

    def fit_params(self, scores_val):
        if self.dist is None:
            self.auto_hpo(scores_val)
        else:
            self.params = self.dist.fit(scores_val.cpu())
            try:
                res = cramervonmises(scores_val.cpu(), self.dist_name, self.params)
                log.info(
                    f"Fitting {str(self.__class__).split('.')[-1][:-2]} p={res.pvalue:.5f}"
                )
            except Exception as e:
                log.info(f"Can not compute p value")
                pass


    def auto_hpo(
        self,
        scores_val,
        candidate_dists=None,
    ):
        if candidate_dists is None:
            candidate_dists = [
                "alpha",
                "genhyperbolic",
                "invweibull",
                "johnsonsb",
                "genextreme",
            ]
        import numpy as np

        best_p = -np.inf
        best_params = None
        best_dist = None
        best_name = None

        for candidate in candidate_dists:
            try:
                cand_dist = getattr(scipy.stats, candidate)
                cand_params = cand_dist.fit(scores_val.cpu())
                res = cramervonmises(scores_val.cpu(), candidate, cand_params)
                p_val = res.pvalue
                log.info(f"Candidate {candidate} yielded p-value {p_val:.5f}")

                if p_val > best_p:
                    best_p = p_val
                    best_params = cand_params
                    best_dist = cand_dist
                    best_name = candidate
            except Exception as e:
                log.warning(f"Fitting {candidate} failed: {e}")
                continue

        if best_dist is not None:
            self.dist = best_dist
            self.dist_name = best_name
            self.params = best_params
            log.info(
                f"Automatic HPO selected distribution {best_name} with p-value {best_p:.5f}"
            )
        else:
            log.error("Automatic HPO failed to select a suitable distribution.")

    @torch.no_grad()
    def predict(self, cfg, data):
        predictions = get_predictions(data, self.mln_attributes)
        mln_p = self.mln.energy(predictions.to(cfg.device)).cpu().squeeze()
        second_scores = self.second_predict(cfg, data)

        if self.dist:
            p_values = self.dist.sf(second_scores, *self.params)
        else:
            p_values = second_scores

        return -mln_p * p_values


def ensemble_score(data, attributes):
    scores = (
        -torch.stack(
            [
                data[f"{att}-logits"].softmax(dim=1).max(dim=1).values
                for att in attributes
            ],
            dim=1,
        )
        .mean(dim=1)
        .double()
        .cpu()
    )
    return scores


class MLNEnsembleDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        name="MLN+Ensemble",
        ensemble_attributes=None,
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_ensemble, mln_attributes=mln_attributes, **kwargs
        )

        if ensemble_attributes is None:
            ensemble_attributes = ["label", "color", "shape"]

        self.ensemble_attributes = ensemble_attributes

    def fit(self, cfg, fitting_data):
        # log.info(f"Fitting {self.name} on {self.ensemble_attributes}")
        scores = ensemble_score(fitting_data, self.ensemble_attributes)
        self.fit_params(scores)

    def _predict_ensemble(self, cfg, data):
        scores = ensemble_score(data, self.ensemble_attributes)
        return scores


class MLNViMDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        vim_detector: ViMDetector,
        name="MLN+ViM",
        attribute="label",
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_vim, mln_attributes=mln_attributes, **kwargs
        )
        self.vim_detector = vim_detector
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name}")

        scores_val = batch_predict_features(
            self.vim_detector.detector,
            fitting_data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        ).cpu()
        self.fit_params(scores_val)

    def _predict_vim(self, cfg, in_data_raw):
        return self.vim_detector.detector.predict_features(
            in_data_raw[f"{self.attribute}-features"].to(cfg.device)
        ).cpu()


class MLNDICEDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        dice_detector: DICEDetector,
        name="MLN+DICE",
        attribute="label",
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_dice, mln_attributes=mln_attributes, **kwargs
        )
        self.dice_detector = dice_detector
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name}")

        scores_val = batch_predict_features(
            self.dice_detector.detector,
            fitting_data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        ).cpu()
        self.fit_params(scores_val)

    def _predict_dice(self, cfg, data):
        return batch_predict_features(
            self.dice_detector.detector,
            data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        )


class MLNSHEDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        she_detector: SHEDetector,
        name="MLN+SHE",
        attribute="label",
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_she, mln_attributes=mln_attributes, **kwargs
        )
        self.she_detector = she_detector
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name}")

        scores_val = batch_predict_features(
            self.she_detector.detector,
            fitting_data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        ).cpu()
        self.fit_params(scores_val)

    def _predict_she(self, cfg, data):
        return batch_predict_features(
            self.she_detector.detector,
            data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        )


class MLNReActDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        react_detector: ReActDetector,
        name="MLN+ReAct",
        attribute="label",
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_she, mln_attributes=mln_attributes, **kwargs
        )
        self.react_detector = react_detector
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name}")

        scores_val = batch_predict_features(
            self.react_detector.detector,
            fitting_data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        ).cpu()
        self.fit_params(scores_val)

    def _predict_she(self, cfg, data):
        return batch_predict_features(
            self.react_detector.detector,
            data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        )


class MLNMahalanobisDetector(MLNCombinedDetector):
    def __init__(
        self,
        mln,
        maha_detector: MahalanobisDetector,
        name="MLN+Mahalanobis",
        attribute="label",
        mln_attributes=None,
        **kwargs,
    ):
        super().__init__(
            name, mln, self._predict_maha, mln_attributes=mln_attributes, **kwargs
        )
        self.maha_detector = maha_detector
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        log.info(f"Fitting {self.name} on {self.attribute}")

        scores_val = batch_predict_features(
            self.maha_detector.detector,
            fitting_data[f"{self.attribute}-features"],
            1024,
            cfg.device,
        ).cpu()
        self.fit_params(scores_val)

    def _predict_maha(self, cfg, in_data_raw):
        return batch_predict_features(
            self.maha_detector.detector,
            in_data_raw[f"{self.attribute}-features"],
            1024,
            cfg.device,
        )


class MLNEBODetector(MLNCombinedDetector):
    def __init__(
        self, mln, name="MLN+EBO", attribute="label", mln_attributes=None, **kwargs
    ):
        super().__init__(name, mln, self._predict_ebo, mln_attributes, **kwargs)
        self.ebo = EnergyBased(model=None)
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        scores_val = batch_predict_features(
            self.ebo, fitting_data[f"{self.attribute}-logits"], 1024, cfg.device
        ).cpu()
        self.fit_params(scores_val)

    def _predict_ebo(self, cfg, in_data_raw):
        return self.ebo.predict_features(in_data_raw[f"{self.attribute}-logits"]).cpu()


class MLNMSPDetector(MLNCombinedDetector):
    def __init__(
        self, mln, name="MLN+MSP", attribute="label", mln_attributes=None, **kwargs
    ):
        super().__init__(
            name, mln, self._predict_msp, mln_attributes=mln_attributes, **kwargs
        )
        self.msp = MaxSoftmax(model=None)
        self.attribute = attribute

    def fit(self, cfg, fitting_data):
        scores_val = batch_predict_features(
            self.msp, fitting_data[f"{self.attribute}-logits"], 1024, cfg.device
        ).cpu()
        self.fit_params(scores_val)

    def _predict_msp(self, cfg, in_data_raw):
        return self.msp.predict_features(in_data_raw[f"{self.attribute}-logits"]).cpu()


class LogicEnsembleDetector(Detector):
    def __init__(
        self,
        constraints,
        name="Logic+Ensemble",
        mln_attributes=None,
        ensemble_attributes=None,
    ):

        super().__init__(name=name)
        if ensemble_attributes is None:
            ensemble_attributes = ["label", "color", "shape"]
        if mln_attributes is None:
            mln_attributes = ["label", "color", "shape", "oe"]
        self.constraints = constraints
        self.mln_attributes = mln_attributes
        self.ensemble_attributes = ensemble_attributes

    def predict(self, cfg, in_data_raw):
        data = torch.stack(
            [
                in_data_raw[f"{att}-logits"].max(dim=1).indices
                for att in self.mln_attributes
            ],
            dim=1,
        )
        valid = (
            torch.stack([f(data) for f in self.constraints], dim=1)
            .all(dim=1)
            .float()
            .squeeze()
        )

        ensemble_scores = torch.stack(
            [
                in_data_raw[f"{att}-logits"].softmax(dim=1).max(dim=1).values
                for att in self.ensemble_attributes
            ],
            dim=1,
        ).mean(dim=1)
        return -valid * ensemble_scores
