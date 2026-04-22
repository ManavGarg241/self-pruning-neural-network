"""
Self-pruning neural network training script for CIFAR-10.
Dataset: CIFAR-10 (3x32x32 color images, 10 classes).
Sparsity strengths tested: lambda values 1e-5, 1e-3, and 1e-1.
Run: python prunable_network.py
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PrunableLinear(nn.Module):
    """Linear layer with trainable sigmoid gates over each weight."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.zeros_(self.gate_scores)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        1) Convert unconstrained gate_scores to (0, 1) using sigmoid.
        2) Apply gates element-wise to weights to get pruned_weights.
        3) Run a standard linear transform with the gated weights and bias.
        """
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(input_tensor, pruned_weights, self.bias)


class SelfPruningNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_all_gates(self) -> torch.Tensor:
        gate_tensors: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gate_tensors.append(torch.sigmoid(module.gate_scores).reshape(-1))
        return torch.cat(gate_tensors, dim=0)


def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    gates = model.get_all_gates()
    return gates.sum()


def get_cifar10_loaders(batch_size: int = 128, pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    device: Union[str, torch.device],
) -> float:
    model.train()
    running_loss = 0.0

    for data, labels in loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(data)
        ce_loss = F.cross_entropy(logits, labels)
        sparse_loss_val = sparsity_loss(model)
        total_loss = ce_loss + lambda_sparse * sparse_loss_val
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / max(len(loader), 1)


def evaluate(model: SelfPruningNet, loader: DataLoader, device: Union[str, torch.device]) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return (100.0 * correct / total) if total > 0 else 0.0


def report_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> float:
    gates = model.get_all_gates()
    num_pruned = (gates < threshold).sum().item()
    total = gates.numel()
    sparsity_pct = (100.0 * num_pruned / total) if total > 0 else 0.0
    print(f"Sparsity: {sparsity_pct:.1f}% of weights pruned.")
    return sparsity_pct


def plot_gate_distribution(model: SelfPruningNet, lambda_val: float) -> None:
    gates = model.get_all_gates().detach().cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=100, range=(0, 1))
    plt.yscale("log")
    plt.xlim(0, 1)
    plt.title(f"Gate Value Distribution (λ={lambda_val})")
    plt.xlabel("Gate value")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=200)
    plt.show()


def run_experiment(
    lambda_sparse: float,
    epochs: int = 20,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    use_cuda = device.type == "cuda"
    train_loader, test_loader = get_cifar10_loaders(pin_memory=use_cuda)
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Using device: {device}")
    if use_cuda:
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lambda_sparse, device)
        print(f"Lambda={lambda_sparse:.0e} | Epoch {epoch + 1}/{epochs} | Avg loss: {avg_loss:.4f}")

    test_accuracy = evaluate(model, test_loader, device)
    sparsity_pct = report_sparsity(model)

    return {
        "lambda_val": lambda_sparse,
        "test_accuracy": test_accuracy,
        "sparsity_pct": sparsity_pct,
        "model": model,
    }


def main() -> None:
    lambda_values = [1e-5, 1e-3, 1e-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for lambda_sparse in lambda_values:
        print("=" * 72)
        print(f"Running experiment with lambda={lambda_sparse:.0e}")
        result = run_experiment(lambda_sparse=lambda_sparse, device=device)
        results.append(result)

    print("\nResults")
    print(f"{'Lambda':<12}{'Test Accuracy (%)':<20}{'Sparsity (%)':<15}")
    for result in results:
        print(
            f"{result['lambda_val']:<12.0e}{result['test_accuracy']:<20.2f}{result['sparsity_pct']:<15.2f}"
        )

    best_model_result = next(r for r in results if abs(r["lambda_val"] - 1e-3) < 1e-12)
    print(f"\nBest model (medium lambda): {best_model_result['lambda_val']:.0e}")
    plot_gate_distribution(best_model_result["model"], best_model_result["lambda_val"])


if __name__ == "__main__":
    main()
