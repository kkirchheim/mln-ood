from collections import deque
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn
from dataset import attribute_index_map, category_value_map

from compiler import ConstraintCompiler


def wrap_shared_model(net):
    """
    In a shared model, we use the weights and biases of the traffic sign label predicting model
    which is fc0
    """

    class MyModel(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.fc = net.fc0

        def forward(self, x):
            z = self.net.features(x)
            return self.fc(z)

    return MyModel(net)


class AccuracyMeter:
    def __init__(self, window_size=50):
        """
        Initializes the accuracy meter with a specified window size.

        Parameters:
        - window_size (int): The number of batches to keep in the moving window.
        """
        self.window_size = window_size
        self.correct_window = deque(maxlen=window_size)
        self.total_window = deque(maxlen=window_size)

    def update(self, preds, labels):
        """
        Updates the moving window with the latest batch's predictions and labels.

        Parameters:
        - preds (torch.Tensor): Predicted labels.
        - labels (torch.Tensor): True labels.
        """
        # Calculate the number of correct predictions
        correct_count = (preds == labels).sum().item()
        total_count = labels.size(0)

        # Update the deque with the new values
        self.correct_window.append(correct_count)
        self.total_window.append(total_count)

    def compute(self):
        """
        Computes the accuracy over the current window.

        Returns:
        - (float): Accuracy over the current window.
        """
        # Sum the values in the window to get total correct and total instances
        window_correct = sum(self.correct_window)
        window_total = sum(self.total_window)

        # Calculate accuracy, handling the case where window_total is zero
        return window_correct / window_total if window_total > 0 else 0.0

    def reset(self):
        """
        Resets the accuracy meter.
        """
        self.correct_window.clear()
        self.total_window.clear()


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, num_classes):
        super(ViTFeatureExtractor, self).__init__()
        self.trafo = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.trafo.heads[-1] = nn.Linear(768, num_classes)

    def features(self, x):
        x = self.trafo._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.trafo.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.trafo.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.trafo._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.trafo.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.trafo.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.trafo.heads(x)

        return x


def get_constraints(cfg):
    compiler = ConstraintCompiler(attribute_index_map, category_value_map)

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

        # Human-friendly constraint:
        # "class=stop_sign -> shape=octagon and color=red"
        # translates to: if x[:, IDX_CLASS]==1 then x[:, IDX_SHAPE]==1 and x[:, IDX_COLOR]==0
        # constraint_str = "class=speed_limit_20 -> (shape=octagon and color=red)"
        generated_code = compiler.compile(constraint_name, constraint_str)

        # Execute the generated code within this namespace
        exec(generated_code, global_namespace, namespace)

    rule_fn = list(namespace.values())
    return rule_fn
