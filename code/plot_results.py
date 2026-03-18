import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ensure_is_member(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure binary label column exists as 0/1."""
    if "is_member" in df.columns:
        df["is_member"] = df["is_member"].astype(int)
        return df

    if "membership" not in df.columns:
        raise ValueError("Input must contain either 'is_member' or 'membership' column.")

    df["is_member"] = (df["membership"].astype(str).str.lower() == "member").astype(int)
    return df


def compute_roc(y_true: pd.Series, y_score: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
    """Compute ROC points and AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc_value


def plot_overall_roc(df: pd.DataFrame, score_col: str, out_dir: Path, title_prefix: str = "") -> float:
    """Plot overall ROC curve for a score column."""
    fpr, tpr, auc_value = compute_roc(df["is_member"], df[score_col])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.2, 5.8))

    plt.plot(
        fpr,
        tpr,
        color="#d95f02",
        lw=2.8,
        label=f"ROC Curve (AUC = {auc_value:.4f})",
    )
    plt.plot([0, 1], [0, 1], color="#666666", lw=1.8, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plot_title = f"{title_prefix} ROC Curve".strip()
    plt.title(plot_title if plot_title else "ROC Curve", fontsize=14, pad=10)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=11)
    plt.grid(alpha=0.22, linewidth=0.8)
    plt.tight_layout()

    safe_name = score_col.replace("lumia_score", "SERSEM")
    pdf_out = out_dir / f"roc_overall_{safe_name}.pdf"
    png_out = out_dir / f"roc_overall_{safe_name}.png"
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=350, bbox_inches="tight")
    plt.close()
    return auc_value


def plot_roc_per_language(df: pd.DataFrame, score_col: str, out_dir: Path, title_prefix: str = "") -> None:
    """Plot ROC curves per language in a single figure."""
    if "language" not in df.columns:
        return

    plt.figure(figsize=(8, 6.2))
    has_any_curve = False

    for language, sub_df in sorted(df.groupby("language"), key=lambda x: str(x[0])):
        # Skip degenerate groups that contain only one class.
        if sub_df["is_member"].nunique() < 2:
            continue
        fpr, tpr, auc_value = compute_roc(sub_df["is_member"], sub_df[score_col])
        plt.plot(fpr, tpr, lw=2.0, label=f"{language} (AUC = {auc_value:.4f}, n={len(sub_df)})")
        has_any_curve = True

    if not has_any_curve:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], color="tab:gray", lw=1.2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plot_title = f"{title_prefix} ROC by Language".strip()
    plt.title(plot_title if plot_title else "ROC by Language")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    safe_name = score_col.replace("lumia_score", "SERSEM")
    pdf_out = out_dir / f"roc_by_language_{safe_name}.pdf"
    png_out = out_dir / f"roc_by_language_{safe_name}.png"
    plt.savefig(pdf_out)
    plt.savefig(png_out, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready ROC plots from eval results."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to eval results file (.csv or .parquet).",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="lumia_score",
        help="Score column to evaluate (default: lumia_score).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="StarCoder2-3B",
        help="Prefix used in figure titles (default: StarCoder2-3B).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path)
    if input_path.suffix.lower() == ".csv":
        results_df = pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".parquet":
        results_df = pd.read_parquet(input_path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet input.")

    results_df = ensure_is_member(results_df)
    if args.score_col not in results_df.columns:
        raise ValueError(f"Column '{args.score_col}' was not found in {input_path}.")

    # Replace NaN scores with 0 only if needed, preserving most values untouched.
    if results_df[args.score_col].isna().any():
        results_df[args.score_col] = results_df[args.score_col].fillna(0.0)

    overall_auc = plot_overall_roc(
        results_df,
        score_col=args.score_col,
        out_dir=output_dir,
        title_prefix=args.title_prefix,
    )
    plot_roc_per_language(
        results_df,
        score_col=args.score_col,
        out_dir=output_dir,
        title_prefix=args.title_prefix,
    )

    print(f"[Done] Input: {input_path}")
    print(f"[Done] Score column: {args.score_col}")
    print(f"[Done] Overall AUC: {overall_auc:.6f}")
    print(f"[Done] Plots saved in: {output_dir.resolve()}")
