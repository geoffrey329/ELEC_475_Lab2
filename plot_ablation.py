"""
ELEC 475: Lab 2 - Master Ablation Summary Plot
Reads the per-epoch validation accuracy of each Table 7.3 model from its training log
(written by train.py) and plots them together with utils.visualization.plot_ablation_summary.

Usage:
    python plot_ablation.py                         # CIFAR-10  -> plots/ablation_summary.png
    python plot_ablation.py --dataset tiny_imagenet  # Tiny IN   -> plots/ablation_summary_tiny.png
"""
import os
import re
import argparse

from utils.visualization import plot_ablation_summary

# Table 7.3 row label -> training log for each dataset
ABLATION_LOGS = {
    "cifar10": {
        "1. Base LeNet-5": "train_lenet_t4.log",
        "2. Modernized LeNet": "train_lenet_modern_t4.log",
        "3. AlexNet (No Norm)": "train_alexnet_none.log",
        "4. AlexNet + LRN": "train_alexnet_lrn.log",
        "5. AlexNet + BatchNorm": "train_alexnet_bn.log",
        "6. AlexNet + BN + Aug": "train_alexnet_with_aug.log",
        "7. PyTorch Zoo AlexNet": "train_zoo_alexnet.log",
    },
    "tiny_imagenet": {
        "1. Base LeNet-5": "train_lenet_tiny.log",
        "2. Modernized LeNet": "train_lenet_modern_tiny.log",
        "3. AlexNet (No Norm)": "train_alexnet_none_tiny.log",
        "4. AlexNet + LRN": "train_alexnet_lrn_tiny.log",
        "5. AlexNet + BatchNorm": "train_alexnet_bn_tiny.log",
        "6. AlexNet + BN + Aug": "train_alexnet_tiny_imagenet.log",
        "7. PyTorch Zoo AlexNet": "train_zoo_alexnet_tiny.log",
    },
}

# Epoch rows of train.py's summary table:
#  Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Time | Throughput
EPOCH_ROW = re.compile(r"^\s*\d+\s*\|")


def read_val_accs(log_path: str):
    """Returns the per-epoch validation accuracies (%) from a train.py log."""
    val_accs = []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if EPOCH_ROW.match(line):
                val_accs.append(float(line.split("|")[4]))
    return val_accs


def main():
    parser = argparse.ArgumentParser(description="ELEC 475: Master Ablation Summary Plot")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"])
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory containing train.py logs")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    out = args.out or ("./plots/ablation_summary.png" if args.dataset == "cifar10"
                       else "./plots/ablation_summary_tiny.png")

    results = {}
    for label, log_name in ABLATION_LOGS[args.dataset].items():
        path = os.path.join(args.log_dir, log_name)
        if not os.path.exists(path):
            print(f"Skipping '{label}': {path} not found")
            continue
        val_accs = read_val_accs(path)
        results[label] = {"val_accs": val_accs}
        print(f"{label:<26} {len(val_accs):>2} epochs | best val acc {max(val_accs):.2f}%")

    plot_ablation_summary(results, save_path=out)


if __name__ == "__main__":
    main()
