"""
LUMIA: Linear probe-based Utilization of Model Internal Activations

This implementation is based on the paper:
"LUMIA: Linear Probes for Membership Inference Attacks Leveraging Internal Activations"

The approach uses Linear Probes (MLPs) trained on internal model activations
to detect whether a sample was part of the model's training data.
"""

import os


import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
import pickle
import re

from MIAttack import MIAttack
from ASTExtractor import ASTExtractor
from LinterExtractor import LinterExtractor
from transformers_compat import tokenize_with_truncation

# Don't override CUDA_VISIBLE_DEVICES - let the user/run script control it
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Commented out for flexibility



class LinearProbe(nn.Module):
    """
    Multi-Layer Perceptron classifier for membership inference.
    Takes layer activations as input and predicts membership probability.

    Simplified 2-layer architecture to reduce overfitting.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class LumiaAttack(MIAttack):
    """
    LUMIA: Uses internal model activations with Linear Probes for MIA.

    Key steps:
    1. Extract activations from each transformer layer
    2. Train an MLP probe per layer to predict membership
    3. Select the layer with highest AUC
    4. Use that layer's probe for final predictions
    """

    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)

        # LUMIA-specific parameters
        self.hidden_dim = getattr(args, 'lumia_hidden_dim', 128)
        self.learning_rate = getattr(args, 'lumia_lr', 1e-3)
        self.epochs = getattr(args, 'lumia_epochs', 100)
        self.batch_size = getattr(args, 'lumia_batch_size', 32)
        self.train_fraction = getattr(args, 'lumia_train_fraction', 0.8)
        self.ensemble_size = getattr(args, 'lumia_ensemble_size', 5)  # Top-K layers to ensemble

        # Cache directory for storing probes and activations
        self.cache_dir = Path(getattr(args, 'lumia_cache_dir', 'my_local_cache/lumia'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model layer information
        self.num_layers = self._get_num_layers()
        self.activation_dim = None

        # Trained probes (one per layer)
        self.probes: Dict[int, LinearProbe] = {}
        self.best_layer = None
        self.ensemble_layers = []  # Top-K layers for ensemble. We have used 5

        # Scaling Weights for Continuous Anomaly Grading
        self.W_BOILERPLATE = 0.1
        self.W_STANDARD = 1.0
        self.W_FORMATTING_ERROR = 5.0
        self.W_REGEX_TARGET = 10.0
        
        self.ast_extractor = ASTExtractor()
        self.linter_extractor = LinterExtractor()
        
        # Psychological Overrides
        self.regex_psych = re.compile(r'(TODO|FIXME|HACK|Note to self)', re.IGNORECASE)
        self.regex_identifier_fb = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{3,}\b')
        self.current_token_weights = None

        print(f"[LUMIA] Initialized with {self.num_layers} layers")

    @property
    def name(self) -> str:
        return "lumia"

    def generate_char_weight_mask(self, text: str, language: str = None) -> np.ndarray:
        # Step 1: Base Character Weight Masking via AST and Dictionaries
        char_weights = self.ast_extractor.extract_weights(text, language)

        # Step 2: Psychological / Inconsistency Overrides
        for match in self.regex_psych.finditer(text):
            char_weights[match.start():match.end()] = self.W_REGEX_TARGET
            
        # Step 3: Offline Linter Diffing
        errors = self.linter_extractor.extract_errors(text, language)
        lines = text.splitlines(keepends=True)
        for (line_num, col_num) in errors:
            if line_num <= len(lines):
                # Calculate absolute character offset
                offset = sum(len(lines[i]) for i in range(line_num - 1)) + (col_num - 1)
                start = max(0, offset - 2)
                end = min(len(text), offset + 3)
                # Boost gracefully
                mask = char_weights[start:end] < self.W_FORMATTING_ERROR
                char_weights[start:end][mask] = self.W_FORMATTING_ERROR

        return char_weights

    def _get_num_layers(self) -> int:
        """Determine the number of transformer layers in the model."""
        # Try different model architectures
        if hasattr(self.model, 'transformer'):
            # GPT-like models
            if hasattr(self.model.transformer, 'h'):
                return len(self.model.transformer.h)
        elif hasattr(self.model, 'model'):
            # Some models wrap the transformer in a 'model' attribute
            if hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)

        # Fallback: try to find layers
        for name, module in self.model.named_modules():
            if 'layers' in name or 'blocks' in name:
                if isinstance(module, nn.ModuleList):
                    return len(module)

        # Default fallback
        print("[LUMIA] Warning: Could not determine number of layers, using 24 as default")
        return 24

    def _extract_activations(self, texts: List[str]) -> Dict[int, np.ndarray]:
        """
        Extract internal activations from all layers for given texts.

        Returns:
            Dictionary mapping layer_idx -> activations array of shape (n_samples, hidden_dim)
        """
        activations_per_layer = {i: [] for i in range(self.num_layers)}

        # Hook function to capture activations
        activation_storage = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]

                # Weighted Average over sequence length using heuristics, or fallback to mean
                if getattr(self, 'current_token_weights', None) is not None:
                    # current_token_weights shape: (seq_len,)
                    weights = self.current_token_weights.unsqueeze(0).unsqueeze(-1).to(output.device) # (1, seq_len, 1)
                    weighted_activation = (output * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
                    mean_activation = output.mean(dim=1)
                    # Concatenate to give the probe both signals
                    comb_activation = torch.cat([mean_activation, weighted_activation], dim=-1)
                else:
                    mean_activation = output.mean(dim=1)
                    # Duplicate to keep shape consistent if weights are missing
                    comb_activation = torch.cat([mean_activation, mean_activation], dim=-1)
                    
                activation_storage[layer_idx] = comb_activation.detach().cpu()
            return hook

        # Register hooks on transformer layers
        hooks = []
        layer_modules = self._get_layer_modules()

        for layer_idx, layer_module in enumerate(layer_modules[:self.num_layers]):
            hook = layer_module.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        # Process texts
        print("[LUMIA] Extracting activations from all layers...")
        for text in tqdm(texts, desc="Extracting activations"):
            try:
                encodings = tokenize_with_truncation(
                    self.tokenizer,
                    text,
                    max_length=self.args.max_length,
                    return_offsets_mapping=True
                )
                
                # Filter out zero length tokens entirely to avoid math crashes on sum
                if encodings.input_ids.size(1) == 0:
                     raise ValueError("Zero length input")
                
                # Compute token weights if offset_mapping is available for the given input
                if "offset_mapping" in encodings:
                    offsets = encodings.pop("offset_mapping")[0].cpu().numpy()
                    seq_len = encodings.input_ids.size(1)
                    
                    char_weights = self.generate_char_weight_mask(text)
                    token_weight_array = np.full(seq_len, self.W_BOILERPLATE, dtype=float)
                    
                    for idx, (start, end) in enumerate(offsets):
                        if start == end: 
                            continue
                        
                        span_weight = np.max(char_weights[start:end]) if start < end else self.W_STANDARD
                        
                        if span_weight <= self.W_STANDARD:
                            token_str = text[start:end].strip()
                            if len(token_str) > 2 and self.regex_identifier_fb.match(token_str):
                                token_weight_array[idx] = self.W_STANDARD
                            else:
                                token_weight_array[idx] = self.W_BOILERPLATE 
                        else:
                            token_weight_array[idx] = span_weight
                    
                    weights_tensor = torch.tensor(token_weight_array, dtype=torch.float32, device=self.model.device)
                    w_max = weights_tensor.max()
                    self.current_token_weights = weights_tensor / (w_max + 1e-8)
                else:
                    self.current_token_weights = None

                inputs = encodings.to(self.model.device)

                with torch.no_grad():
                    # Forward pass triggers hooks
                    _ = self.model(**inputs)

                # Store activations from this sample
                for layer_idx in range(self.num_layers):
                    if layer_idx in activation_storage:
                        activations_per_layer[layer_idx].append(
                            activation_storage[layer_idx].numpy()
                        )

                activation_storage.clear()

            except Exception as e:
                print(f"[LUMIA] Error extracting activations: {e}")
                for layer_idx in range(self.num_layers):
                    if self.activation_dim is None and len(activations_per_layer[layer_idx]) > 0:
                        self.activation_dim = activations_per_layer[layer_idx][0].shape[-1]
                    dim = self.activation_dim if self.activation_dim else self.hidden_dim * 2
                    activations_per_layer[layer_idx].append(
                        np.zeros((1, dim), dtype=np.float32)
                    )

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Convert to numpy arrays
        result = {}
        for layer_idx in range(self.num_layers):
            if activations_per_layer[layer_idx]:
                # Shape: (n_samples, hidden_dim)
                result[layer_idx] = np.vstack(activations_per_layer[layer_idx])
                if self.activation_dim is None:
                    self.activation_dim = result[layer_idx].shape[-1]

        return result

    def _get_layer_modules(self) -> List[nn.Module]:
        """Get list of transformer layer modules."""
        # Try different model architectures
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                return list(self.model.transformer.h)
        elif hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                return list(self.model.model.layers)

        # Fallback: collect all layer-like modules
        layers = []
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                if isinstance(module, nn.Module) and not isinstance(module, nn.ModuleList):
                    layers.append(module)

        return layers[:self.num_layers] if layers else []

    def _train_probe(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        layer_idx: int
    ) -> Tuple[LinearProbe, float]:
        """
        Train a linear probe for a specific layer.

        The val split passed here is used ONLY for early stopping and
        best-weights selection. It is intentionally kept separate from the
        test split that train_probes() uses for layer selection and ensemble
        construction, so the two decisions never share the same data.

        Returns:
            Trained probe and validation AUC.
        """
        from sklearn.metrics import roc_auc_score

        probe = LinearProbe(self.activation_dim, self.hidden_dim).to(self.model.device)
        # Add L2 regularization (weight_decay) to prevent overfitting
        optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # L2 regularization penalty
        )
        criterion = nn.BCELoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.model.device)
        y_train_t = torch.FloatTensor(y_train).to(self.model.device)
        X_val_t = torch.FloatTensor(X_val).to(self.model.device)
        y_val_t = torch.FloatTensor(y_val).to(self.model.device)

        best_auc = 0
        best_state = None
        patience = 5  
        patience_counter = 0

        for epoch in range(self.epochs):
            probe.train()

            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(X_train_t), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train_t[batch_indices]
                y_batch = y_train_t[batch_indices]

                optimizer.zero_grad()
                predictions = probe(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Validation 
            if epoch % 5 == 0:
                probe.eval()
                with torch.no_grad():
                    val_predictions = probe(X_val_t).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_predictions)

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = probe.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        # Restore best model
        if best_state is not None:
            probe.load_state_dict(best_state)

        return probe, best_auc

    def train_probes(self, texts: List[str], labels: List[int]):
        """
        Train linear probes on all layers and select the best one.

        Uses a strict 3-way split to prevent validation set double-dipping:
          - train  (60%) → weight updates inside _train_probe
          - val    (20%) → early stopping + best-weights selection ONLY
          - test   (20%) → layer selection + ensemble construction ONLY

        The key invariant: _train_probe never sees test_indices, so the
        test AUCs used for layer selection are unbiased held-out estimates.

        Args:
            texts: List of text samples
            labels: List of membership labels (1 = member, 0 = non-member)
        """
        print(f"[LUMIA] Training probes on {len(texts)} samples...")

        # Extract activations from all layers
        activations = self._extract_activations(texts)
        labels = np.array(labels)

        # ── 3-way split ──────────────────────────────────────────────────────
        #
        #   train  → _train_probe weight updates
        #   val    → early stopping / best-weights selection (inside _train_probe)
        #   test   → layer selection + ensemble construction (here, after training)
        #
        n_samples = len(labels)
        # ── Stratified 3-way split ──────────────────────────────────────────
        # We need to ensure that train, val, and test splits all have the same
        # member/non-member ratio (50/50).
        
        train_indices = []
        val_indices   = []
        test_indices  = []
        
        for label_val in [0, 1]:
            label_idx = np.where(labels == label_val)[0]
            np.random.shuffle(label_idx)
            
            n_lab = len(label_idx)
            n_lab_train = int(n_lab * 0.60)
            n_lab_val   = int(n_lab * 0.20)
            
            train_indices.extend(label_idx[:n_lab_train])
            val_indices.extend(label_idx[n_lab_train : n_lab_train + n_lab_val])
            test_indices.extend(label_idx[n_lab_train + n_lab_val :])
            
        train_indices = np.array(train_indices)
        val_indices   = np.array(val_indices)
        test_indices  = np.array(test_indices)

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        print(
            f"[LUMIA] Stratified split — "
            f"train: {len(train_indices)} ({np.sum(labels[train_indices])}m / {len(train_indices)-np.sum(labels[train_indices])}nm), "
            f"val: {len(val_indices)} ({np.sum(labels[val_indices])}m / {len(val_indices)-np.sum(labels[val_indices])}nm), "
            f"test: {len(test_indices)} ({np.sum(labels[test_indices])}m / {len(test_indices)-np.sum(labels[test_indices])}nm)"
        )

        # ── Train one probe per layer ─────────────────────────────────────────
        # _train_probe only ever sees train + val splits.
        # test_indices are withheld completely until all probes are trained.
        layer_aucs = {}

        for layer_idx in tqdm(range(self.num_layers), desc="Training probes per layer"):
            if layer_idx not in activations:
                continue

            X = activations[layer_idx]

            probe, val_auc = self._train_probe(
                X_train=X[train_indices],
                y_train=labels[train_indices],
                X_val=X[val_indices],
                y_val=labels[val_indices],
                layer_idx=layer_idx,
            )
            self.probes[layer_idx] = probe

            # ── Evaluate on the clean held-out test split ─────────────────────
            # This AUC drives layer selection and ensemble construction.
            from sklearn.metrics import roc_auc_score
            probe.eval()
            with torch.no_grad():
                X_test_t = torch.FloatTensor(X[test_indices]).to(self.model.device)
                test_preds = probe(X_test_t).cpu().numpy()
            test_auc = roc_auc_score(labels[test_indices], test_preds)

            layer_aucs[layer_idx] = test_auc
            print(
                f"[LUMIA] Layer {layer_idx}: "
                f"val_auc (early-stop only) = {val_auc:.4f}  |  "
                f"test_auc (selection)      = {test_auc:.4f}"
            )

        # ── Layer selection and ensemble construction — test AUCs only ────────
        # Both decisions use the clean test split, never the val split.
        self.best_layer = max(layer_aucs.items(), key=lambda x: x[1])[0]
        best_auc        = layer_aucs[self.best_layer]

        sorted_layers        = sorted(layer_aucs.items(), key=lambda x: x[1], reverse=True)
        self.ensemble_layers = [layer_idx for layer_idx, _ in sorted_layers[: self.ensemble_size]]

        print(f"\n[LUMIA] Best layer: {self.best_layer} with test_auc = {best_auc:.4f}")
        print(f"[LUMIA] Ensemble layers (top {self.ensemble_size}): {self.ensemble_layers}")

        # Save probes
        self._save_probes()

        return train_indices, val_indices, test_indices

    def _save_probes(self):
        """Save trained probes to disk."""
        model_name = self.args.model_name.replace('/', '_')
        probe_file = self.cache_dir / f"probes_{model_name}.pkl"

        save_data = {
            'probes': {k: v.state_dict() for k, v in self.probes.items()},
            'best_layer': self.best_layer,
            'ensemble_layers': self.ensemble_layers,
            'activation_dim': self.activation_dim,
            'num_layers': self.num_layers,
            'train_indices': getattr(self, 'train_indices', None),
            'val_indices': getattr(self, 'val_indices', None),
            'test_indices': getattr(self, 'test_indices', None)
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(probe_file, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[LUMIA] Probes saved to {probe_file}")

    def _load_probes(self) -> bool:
        """Load trained probes from disk if available."""
        model_name = self.args.model_name.replace('/', '_')
        probe_file = self.cache_dir / f"probes_{model_name}.pkl"

        if not probe_file.exists():
            return False

        try:
            with open(probe_file, 'rb') as f:
                save_data = pickle.load(f)

            self.activation_dim = save_data['activation_dim']
            self.best_layer = save_data['best_layer']
            self.ensemble_layers = save_data.get('ensemble_layers', [self.best_layer])  # Backward compat
            
            self.train_indices = save_data.get('train_indices', None)
            self.val_indices = save_data.get('val_indices', None)
            self.test_indices = save_data.get('test_indices', None)

            for layer_idx, state_dict in save_data['probes'].items():
                probe = LinearProbe(self.activation_dim, self.hidden_dim).to(self.model.device)
                probe.load_state_dict(state_dict)
                self.probes[layer_idx] = probe

            print(f"[LUMIA] Loaded probes from {probe_file}")
            print(f"[LUMIA] Best layer: {self.best_layer}")
            print(f"[LUMIA] Ensemble layers: {self.ensemble_layers}")
            return True

        except Exception as e:
            print(f"[LUMIA] Error loading probes: {e}")
            return False

    def compute_scores(self, texts: List[str], languages: List[str] = None) -> List[float]:
        """
        Compute MIA scores for given texts using trained probes.

        Args:
            texts: List of text samples
            languages: List of programming languages (optional, not used by LUMIA)

        If probes are not trained, this will return random scores and print a warning.
        In practice, you should call train_probes() first with labeled data.
        """
        # Try to load pre-trained probes
        if not self.probes:
            if not self._load_probes():
                print("[LUMIA] WARNING: No trained probes found. Please train probes first.")
                print("[LUMIA] Returning random scores as placeholder.")
                return [np.random.random() for _ in texts]

        # Extract activations
        activations = self._extract_activations(texts)

        # Use ensemble of top-K layers if available, otherwise fall back to best layer
        use_ensemble = len(self.ensemble_layers) > 1 and all(
            layer in activations and layer in self.probes for layer in self.ensemble_layers
        )

        if use_ensemble:
            print(f"[LUMIA] Using ensemble of {len(self.ensemble_layers)} layers for prediction")
            # Ensemble prediction: average scores from top-K layers
            all_scores = []

            for layer_idx in self.ensemble_layers:
                X = activations[layer_idx]
                probe = self.probes[layer_idx]
                probe.eval()

                layer_scores = []
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.model.device)
                    for i in range(0, len(X_tensor), self.batch_size):
                        batch = X_tensor[i:i + self.batch_size]
                        batch_scores = probe(batch).cpu().numpy()
                        layer_scores.extend(batch_scores.tolist())

                all_scores.append(np.array(layer_scores))

            # Average predictions across layers
            scores = np.mean(all_scores, axis=0).tolist()
        else:
            # Fallback to single best layer
            print(f"[LUMIA] Using single best layer ({self.best_layer}) for prediction")
            if self.best_layer not in activations or self.best_layer not in self.probes:
                print(f"[LUMIA] WARNING: Best layer {self.best_layer} not available.")
                return [np.random.random() for _ in texts]

            X = activations[self.best_layer]
            probe = self.probes[self.best_layer]
            probe.eval()
            scores = []

            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.model.device)
                for i in range(0, len(X_tensor), self.batch_size):
                    batch = X_tensor[i:i + self.batch_size]
                    batch_scores = probe(batch).cpu().numpy()
                    scores.extend(batch_scores.tolist())

        return scores

    def compute_scores_with_vectors(self, texts: List[str], languages: List[str] = None) -> Tuple[List[float], List[List[float]]]:
        """
        Compute LUMIA scores and per-sample confidence vectors.

        Returns:
            scores: List of scalar membership probabilities (as in compute_scores)
            vectors: List of per-sample vectors, each containing the per-layer
                     probe outputs used to form the final score (e.g., ensemble).
        """
        # Try to load pre-trained probes
        if not self.probes:
            if not self._load_probes():
                print("[LUMIA] WARNING: No trained probes found. Please train probes first.")
                print("[LUMIA] Returning random scores as placeholder.")
                rand_scores = [np.random.random() for _ in texts]
                rand_vectors = [[s] for s in rand_scores]
                return rand_scores, rand_vectors

        # Extract activations
        activations = self._extract_activations(texts)

        # Determine which layers to use
        use_ensemble = len(self.ensemble_layers) > 1 and all(
            layer in activations and layer in self.probes for layer in self.ensemble_layers
        )

        if use_ensemble:
            layer_indices = list(self.ensemble_layers)
        else:
            if self.best_layer not in activations or self.best_layer not in self.probes:
                print(f"[LUMIA] WARNING: Best layer {self.best_layer} not available.")
                rand_scores = [np.random.random() for _ in texts]
                rand_vectors = [[s] for s in rand_scores]
                return rand_scores, rand_vectors
            layer_indices = [self.best_layer]

        # Compute per-layer scores
        per_layer_scores = []
        for layer_idx in layer_indices:
            X = activations[layer_idx]
            probe = self.probes[layer_idx]
            probe.eval()

            layer_scores = []
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.model.device)
                for i in range(0, len(X_tensor), self.batch_size):
                    batch = X_tensor[i:i + self.batch_size]
                    batch_scores = probe(batch).cpu().numpy()
                    layer_scores.extend(batch_scores.tolist())

            per_layer_scores.append(np.array(layer_scores))

        per_layer_scores = np.stack(per_layer_scores, axis=1)  # (n_samples, n_layers_used)

        # Final scalar scores are the mean across used layers
        scores = per_layer_scores.mean(axis=1).tolist()
        vectors = per_layer_scores.tolist()

        return scores, vectors
