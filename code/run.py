# Standard library
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type

# Third-party
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Monkey-patch for transformers 4.52.0 vs torch 2.4+ compatibility
import torch.utils._pytree as pytree
if not hasattr(pytree, 'register_pytree_node') and hasattr(pytree, '_register_pytree_node'):
    def _register_pytree_node_wrapper(typ, flatten_fn, unflatten_fn, **kwargs):
        return pytree._register_pytree_node(typ, flatten_fn, unflatten_fn)
    pytree.register_pytree_node = _register_pytree_node_wrapper

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from MIAttack import MIAttack
from LumiaAttack import LumiaAttack
from transformers_compat import load_model_safe

def load_model_from_directory(model_name: str, args):
    print(f"Loading '{model_name}' (fp16={args.use_fp16}, bf16={getattr(args, 'use_bf16', False)}, force_cpu={getattr(args, 'force_cpu', False)})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device_mapping = "cpu" if getattr(args, 'force_cpu', False) else "auto"
    if getattr(args, 'use_bf16', False):
        torch_dtype = torch.bfloat16
    elif args.use_fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
        
    model = load_model_safe(
        AutoModelForCausalLM,
        model_name,
        device_map=device_mapping,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer

# Attack Registry
ATTACK_REGISTRY: Dict[str, Type[MIAttack]] = {
    "lumia": LumiaAttack,
}

# ============================================================================
# Main Experiment Class
# ============================================================================

class MIAExperiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Initialize model and tokenizer
        print(f"Loading model: {args.model_name}")
        self.model, self.tokenizer = load_model_from_directory("" + args.model_name, args=args)

    def load_datasets(self) -> pd.DataFrame:
        """
        Load member and non-member datasets.

        Ensures each language has exactly `args.samples_per_language` rows,
        with a strict 50/50 split between members and non-members within each language.
        """
        subsets = ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        cap = self.args.samples_per_language
        half_cap = cap // 2

        for subset in subsets:
            print(f"[Dataset] Loading {subset}...")
            ds = load_dataset(self.args.dataset, subset, split="test")
            df = ds.to_pandas()
            df['language'] = subset
            df['is_member'] = (df['membership'] == 'member').astype(int)

            # Separate members and non-members
            members = df[df['is_member'] == 1]
            non_members = df[df['is_member'] == 0]

            print(f"  - Available: {len(members)} members, {len(non_members)} non-members")

            # Sample exactly half_cap from each
            if len(members) < half_cap or len(non_members) < half_cap:
                actual_half = min(len(members), len(non_members), half_cap)
                print(
                    f"  [WARNING] {subset} has insufficient samples for 50/50 split of {cap}. "
                    f"Using {actual_half * 2} samples total ({actual_half} each)."
                )
                members_sampled = members.sample(n=actual_half, random_state=self.args.seed)
                non_members_sampled = non_members.sample(n=actual_half, random_state=self.args.seed)
            else:
                members_sampled = members.sample(n=half_cap, random_state=self.args.seed)
                non_members_sampled = non_members.sample(n=half_cap, random_state=self.args.seed)

            dfs.append(pd.concat([members_sampled, non_members_sampled]))

        full_df = pd.concat(dfs, ignore_index=True)

        print(
            f"[Dataset] Loaded {len(full_df)} samples total."
        )
        for subset in subsets:
            lang_df = full_df[full_df['language'] == subset]
            m = lang_df['is_member'].sum()
            nm = len(lang_df) - m
            print(f"  - {subset}: {len(lang_df)} samples ({m}m / {nm}nm)")

        return full_df

    def save_results(self, df: pd.DataFrame, df_shadow: pd.DataFrame, executed_attacks: List[str]):
        """Save results to disk and create a replication package."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{self.args.model_name.replace('/', '_')}_{timestamp}"
        
        # Create replication package directory
        repo_dir = self.output_dir / f"replication_{exp_id}"
        repo_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Evaluation Results
        output_file = repo_dir / "eval_results.parquet"
        columns_to_keep = ['content', 'membership', 'is_member', 'language', 'split_role']
        for attack_name in executed_attacks:
            score_col = f"{attack_name}_score"
            if score_col in df.columns:
                columns_to_keep.append(score_col)
            vector_col = f"{attack_name}_confidence_vector"
            if vector_col in df.columns:
                columns_to_keep.append(vector_col)
        df[columns_to_keep].to_parquet(output_file, index=False)
        print(f"\n[Replication] Eval results saved to: {output_file}")

        # 2. Save Shadow Dataset (Training data for attacks)
        shadow_file = repo_dir / "shadow_dataset.parquet"
        df_shadow.to_parquet(shadow_file, index=False)
        print(f"[Replication] Shadow dataset saved to: {shadow_file}")

        # 3. Save Metadata
        metadata = {
            'timestamp': timestamp,
            'model_name': self.args.model_name,
            'samples_per_language': self.args.samples_per_language,
            'seed': self.args.seed,
            'max_length': self.args.max_length,
            'attacks_executed': executed_attacks,
            'total_samples_eval': len(df),
            'total_samples_shadow': len(df_shadow),
        }
        metadata_file = repo_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[Replication] Metadata saved to: {metadata_file}")

        # 4. Copy LUMIA probes if they exist
        if "lumia" in executed_attacks:
            lumia_cache = Path(self.args.lumia_cache_dir)
            probe_name = f"probes_{self.args.model_name.replace('/', '_')}.pkl"
            src_probe = lumia_cache / probe_name
            if src_probe.exists():
                import shutil
                shutil.copy2(src_probe, repo_dir / probe_name)
                print(f"[Replication] LUMIA probes copied to package: {repo_dir / probe_name}")
            else:
                print(f"[WARNING] LUMIA probe {src_probe} not found, could not copy to replication package.")

        # 5. Archive Source Code for reproduction
        code_dir = repo_dir / "code"
        code_dir.mkdir(exist_ok=True)
        files_to_copy = [
            "run.py", "LumiaAttack.py", "MIAttack.py", "ASTExtractor.py", 
            "LinterExtractor.py", "WeightedAnomalyMIA.py", "NeighborhoodAttack.py",
            "MetaEnsembleAttack.py"
        ]
        import shutil
        cwd = Path.cwd()
        for f_name in files_to_copy:
            src_f = cwd / f_name
            if src_f.exists():
                shutil.copy2(src_f, code_dir / f_name)
        print(f"[Replication] Source code archived in: {code_dir}")

        print(f"\n{'=' * 60}")
        print(f"Replication package created at: {repo_dir}")
        print(f"{'=' * 60}")

    def run(self):
        """Run the complete experiment."""
        print(f"\n{'=' * 60}")
        print("Starting MIA Experiment")
        print(f"{'=' * 60}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Attacks: {', '.join(self.args.attacks)}")
        print(f"{'=' * 60}\n")

        # ── Load data ─────────────────────────────────────────────────────────
        # After load_datasets() the pool is exactly 25 000 rows (5k × 5 languages),
        # balanced across languages and with a known member/non-member ratio.
        full_df = self.load_datasets()

        # ── Shadow / evaluation split ─────────────────────────────────────────
        # We split the pool into two disjoint halves: shadow (train) and eval (test).
        # We use stratified splitting to ensure each language AND membership class
        # is split exactly 50/50 between shadow and eval sets.
        
        shadow_indices = []
        eval_indices = []
        
        probe_exists = False
        if "lumia" in self.args.attacks:
            lumia_cache = Path(self.args.lumia_cache_dir)
            probe_name = f"probes_{self.args.model_name.replace('/', '_')}.pkl"
            if (lumia_cache / probe_name).exists():
                probe_exists = True
        
        for lang in full_df['language'].unique():
            for is_mem in [0, 1]:
                subset_idx = full_df[(full_df['language'] == lang) & (full_df['is_member'] == is_mem)].index.tolist()
                random.shuffle(subset_idx)
                
                if probe_exists:
                    eval_indices.extend(subset_idx)
                else:
                    mid = len(subset_idx) // 2
                    shadow_indices.extend(subset_idx[:mid])
                    eval_indices.extend(subset_idx[mid:])
        
        df_shadow = full_df.loc[shadow_indices].reset_index(drop=True)
        df        = full_df.loc[eval_indices].reset_index(drop=True)

        df_shadow['split_role'] = 'shadow'
        df['split_role'] = 'eval'

        print(f"\n[Split] Shadow  (train): {len(df_shadow)} samples")
        print(f"[Split] Eval    (test) : {len(df)} samples")
        print("[Split] Per-language Shadow Balance:")
        for lang in df_shadow['language'].unique():
            l_df = df_shadow[df_shadow['language'] == lang]
            m = l_df['is_member'].sum()
            nm = len(l_df) - m
            print(f"  - {lang}: {len(l_df)} samples ({m}m / {nm}nm)")
        
        print("[Split] Per-language Eval Balance:")
        for lang in df['language'].unique():
            l_df = df[df['language'] == lang]
            m = l_df['is_member'].sum()
            nm = len(l_df) - m
            print(f"  - {lang}: {len(l_df)} samples ({m}m / {nm}nm)")
        
        print(
            f"[Split] Total Eval class balance — "
            f"members: {df['is_member'].sum()}, "
            f"non-members: {(df['is_member'] == 0).sum()}"
        )

        # ── Execute attacks ───────────────────────────────────────────────────
        executed_attacks = []
        for attack_key in self.args.attacks:
            if attack_key not in ATTACK_REGISTRY:
                print(f"Warning: Attack '{attack_key}' not found in registry. Skipping.")
                continue

            attack_class = ATTACK_REGISTRY[attack_key]
            attacker = attack_class(self.args, self.model, self.tokenizer)

            # Special handling for LUMIA — train probes on shadow data if not cached
            if attack_key == "lumia" and isinstance(attacker, LumiaAttack):
                if not attacker._load_probes():
                    print(f"\n[LUMIA] No pre-trained probes found. Training probes first...")
                    print(f"[LUMIA] Using {len(df_shadow)} samples from shadow dataset for training.")
                    print(f"[LUMIA] This may take 10-30 minutes depending on dataset size.")
                    train_indices, val_indices, test_indices = attacker.train_probes(
                        texts=df_shadow['content'].tolist(),
                        labels=df_shadow['is_member'].tolist(),
                    )
                    df_shadow.loc[train_indices, 'split_role'] = 'shadow_train'
                    df_shadow.loc[val_indices, 'split_role'] = 'shadow_val'
                    df_shadow.loc[test_indices, 'split_role'] = 'shadow_test'
                    print(
                        f"[LUMIA] Probe training complete. "
                        f"Proceeding with inference on {len(df)} held-out eval samples..."
                    )
                else:
                    print(f"\n[LUMIA] Pre-trained probes mapped from cache.")
                    t_idx = getattr(attacker, 'train_indices', None)
                    v_idx = getattr(attacker, 'val_indices', None)
                    test_idx = getattr(attacker, 'test_indices', None)
                    if t_idx is not None:
                        try:
                            df_shadow.loc[t_idx, 'split_role'] = 'shadow_train'
                            df_shadow.loc[v_idx, 'split_role'] = 'shadow_val'
                            df_shadow.loc[test_idx, 'split_role'] = 'shadow_test'
                        except KeyError:
                            df_shadow['split_role'] = 'shadow_cached_out_of_bounds'
                    else:
                        df_shadow['split_role'] = 'shadow_cached_unknown_split'

            print(f"\nRunning attack: {attacker.name}")

            # df is the held-out eval set — labels in df['is_member'] are NOT
            # passed to compute_scores(); the attack only sees the raw content.
            if hasattr(attacker, "compute_scores_with_vectors"):
                scores, vectors = attacker.compute_scores_with_vectors(
                    texts=df['content'].tolist(),
                    languages=df['language'].tolist() if 'language' in df.columns else None,
                )
                df[f"{attacker.name}_score"] = scores
                df[f"{attacker.name}_confidence_vector"] = vectors
            else:
                scores = attacker.compute_scores(
                    texts=df['content'].tolist(),
                    languages=df['language'].tolist() if 'language' in df.columns else None,
                )
                df[f"{attacker.name}_score"] = scores

            executed_attacks.append(attacker.name)

        # ── Report AUC metrics on eval set ─────────────────────────────────────
        if executed_attacks:
            print(f"\n[Metrics] Membership inference performance on eval set")
            for attack_name in executed_attacks:
                score_col = f"{attack_name}_score"
                if score_col not in df.columns:
                    continue
                y_true = df["is_member"]
                y_score = df[score_col]
                try:
                    overall_auc = roc_auc_score(y_true, y_score)
                except ValueError:
                    overall_auc = float("nan")
                print(f"\n  Attack: {attack_name}")
                print(f"    Overall AUC: {overall_auc:.6f}")
                print(f"    AUC per language:")
                for lang, sub in df.groupby("language"):
                    try:
                        lang_auc = roc_auc_score(sub["is_member"], sub[score_col])
                        print(f"      - {lang}: {lang_auc:.4f} (n={len(sub)})")
                    except ValueError:
                        print(f"      - {lang}: AUC undefined (only one class present, n={len(sub)})")

        # ── Save results and create replication package ───────────────────────
        self.save_results(df, df_shadow, executed_attacks)

        print(f"\n{'=' * 60}")
        print("Experiment completed successfully!")
        print(f"{'=' * 60}\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Membership Inference Attack experiments on code datasets"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'JetBrains/Mellum-4b-base')"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
        help="Use FP16 precision for faster inference"
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=False,
        help="Use BF16 precision for faster inference"
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="Force the model to run on CPU only"
    )

    # MKP-specific/utility flags
    parser.add_argument(
        "--use_sliding_window",
        action="store_true",
        default=False,
        help="Use sliding-window probability calculation in Min-K Prob attack"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="AISE-TUDelft/Poisoned-Chalice",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--samples_per_language",
        type=int,
        default=10000,
        help=(
            "Number of rows to sample per language (Go, Java, Python, Ruby, Rust). "
            "Total pool = samples_per_language × 5. Default: 10 000 → 50 000 total, "
            "giving exactly 5 000 per language in each 50/50 shadow/eval half."
        )
    )

    # Attack parameters
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        choices=["lumia"],
        default=["lumia"],
        help="Which attacks to run"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )

    # Other parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # LUMIA-specific parameters
    parser.add_argument(
        "--lumia_hidden_dim",
        type=int,
        default=128,
        help="LUMIA probe hidden dimension"
    )
    parser.add_argument(
        "--lumia_lr",
        type=float,
        default=1e-3,
        help="LUMIA learning rate"
    )
    parser.add_argument(
        "--lumia_epochs",
        type=int,
        default=100,
        help="LUMIA training epochs"
    )
    parser.add_argument(
        "--lumia_batch_size",
        type=int,
        default=32,
        help="LUMIA training batch size"
    )
    parser.add_argument(
        "--lumia_train_fraction",
        type=float,
        default=0.8,
        help="LUMIA train/val split fraction (legacy, no longer used by train_probes)"
    )
    parser.add_argument(
        "--lumia_cache_dir",
        type=str,
        default="my_local_cache/lumia",
        help="LUMIA cache directory for probes"
    )
    parser.add_argument(
        "--lumia_ensemble_size",
        type=int,
        default=5,
        help="LUMIA number of top layers to ensemble (1 = no ensemble, 5 = default)"
    )

    return parser.parse_args()

import multiprocessing

if __name__ == "__main__":
    # Essential for preventing PyTorch / Tokenizer deadlocks on macOS
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    args = parse_args()
    print(args)
    experiment = MIAExperiment(args)
    experiment.run()