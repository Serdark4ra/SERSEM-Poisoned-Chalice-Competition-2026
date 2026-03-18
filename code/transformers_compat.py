"""
Transformers Compatibility Layer
=================================

This module provides compatibility functions to ensure code works across
transformers versions 4.52 and 5.3+.

Key differences addressed:
1. ModelOutput vs tuple returns
2. inputs_embeds handling
3. Tokenizer behavior (including pad_token)
4. generate() API differences
5. from_pretrained parameters
6. Gradient flow in backward-compatible way
"""

import torch
from typing import Any, Dict, Optional, Union
import transformers


def get_transformers_version():
    """Get the transformers version as a tuple (major, minor, patch)."""
    version_str = transformers.__version__
    parts = version_str.split('.')[:3]
    return tuple(int(p) for p in parts)


# Version check
TRANSFORMERS_VERSION = get_transformers_version()
IS_OLD_VERSION = TRANSFORMERS_VERSION[0] == 4 and TRANSFORMERS_VERSION[1] <= 52
IS_NEW_VERSION = TRANSFORMERS_VERSION[0] >= 5


def get_logits(outputs):
    """
    Extract logits from model outputs in a version-agnostic way.

    Args:
        outputs: Model outputs (could be ModelOutput object or tuple)

    Returns:
        logits tensor
    """
    if hasattr(outputs, 'logits'):
        return outputs.logits
    elif isinstance(outputs, tuple):
        return outputs[0]
    else:
        # Fallback: assume it's the logits directly
        return outputs


def get_loss(outputs):
    """
    Extract loss from model outputs in a version-agnostic way.

    Args:
        outputs: Model outputs (could be ModelOutput object or tuple)

    Returns:
        loss tensor or None
    """
    if hasattr(outputs, 'loss'):
        return outputs.loss
    elif isinstance(outputs, tuple) and len(outputs) > 1:
        # Some models return (loss, logits, ...) as tuple
        return outputs[0] if outputs[0].dim() == 0 else None
    else:
        return None


def ensure_pad_token(tokenizer):
    """
    Ensure tokenizer has a pad token (common issue with code models).

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        tokenizer (modified in-place if needed)
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Some models don't have eos_token either, use a special token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def tokenize_with_truncation(tokenizer, text, max_length=None, **kwargs):
    """
    Tokenize text with explicit truncation/padding settings for compatibility.

    Args:
        tokenizer: HuggingFace tokenizer
        text: Input text or list of texts
        max_length: Maximum sequence length
        **kwargs: Additional tokenizer arguments

    Returns:
        Tokenized inputs
    """
    # Ensure tokenizer has a pad token (common issue with code models)
    ensure_pad_token(tokenizer)

    # Set defaults for compatibility
    tokenize_kwargs = {
        'return_tensors': 'pt',
        'padding': True,
        'truncation': True,
    }

    if max_length is not None:
        tokenize_kwargs['max_length'] = max_length

    # Override with user-provided kwargs
    tokenize_kwargs.update(kwargs)

    return tokenizer(text, **tokenize_kwargs)


def model_forward_with_embeds(model, inputs_embeds, attention_mask=None, labels=None, **kwargs):
    """
    Perform forward pass with embeddings in a version-agnostic way.

    Args:
        model: HuggingFace model
        inputs_embeds: Input embeddings tensor
        attention_mask: Optional attention mask
        labels: Optional labels for loss computation
        **kwargs: Additional model arguments

    Returns:
        Model outputs
    """
    forward_kwargs = {
        'inputs_embeds': inputs_embeds,
    }

    if attention_mask is not None:
        forward_kwargs['attention_mask'] = attention_mask

    if labels is not None:
        forward_kwargs['labels'] = labels

    # Add any additional kwargs
    forward_kwargs.update(kwargs)

    return model(**forward_kwargs)


def get_embeddings_with_grad(model, input_ids):
    """
    Get input embeddings with proper gradient tracking for both versions.

    Args:
        model: HuggingFace model
        input_ids: Input token IDs

    Returns:
        Embeddings tensor with requires_grad=True
    """
    # Get embeddings layer
    embeddings_layer = model.get_input_embeddings()

    # Get embeddings and ensure gradient tracking
    inputs_embeds = embeddings_layer(input_ids)

    # Detach and re-enable gradients (more robust across versions)
    inputs_embeds = inputs_embeds.detach()
    inputs_embeds.requires_grad_(True)

    # Ensure we can access gradients (needed in some versions)
    if hasattr(inputs_embeds, 'retain_grad'):
        inputs_embeds.retain_grad()

    return inputs_embeds


def load_model_safe(model_class, model_name, device_map="auto", torch_dtype=None, **kwargs):
    """
    Load model with version-compatible parameters.

    Args:
        model_class: Model class (e.g., AutoModelForCausalLM)
        model_name: Model name or path
        device_map: Device mapping strategy ("auto", "cuda", "cpu", or device map dict)
        torch_dtype: Torch dtype (e.g., torch.float16)
        **kwargs: Additional loading arguments

    Returns:
        Loaded model
    """
    load_kwargs = {}

    # Handle device_map with transformers 4.52 fix
    # In transformers 4.52, device_map="auto" sometimes defaults to CPU
    # We need to be more explicit
    if device_map is not None:
        if device_map == "auto" and IS_OLD_VERSION and torch.cuda.is_available():
            # For transformers 4.52, use "cuda" instead of "auto" if GPU available
            # This ensures the model actually loads on GPU
            load_kwargs['device_map'] = "cuda"
            print(f"[Compat] Using device_map='cuda' instead of 'auto' for transformers {TRANSFORMERS_VERSION}")
        else:
            load_kwargs['device_map'] = device_map

    # Handle torch_dtype - avoid "auto" in old versions
    if torch_dtype is not None:
        if torch_dtype == "auto" and IS_OLD_VERSION:
            # Fall back to float32 for old versions
            load_kwargs['torch_dtype'] = torch.float32
        else:
            load_kwargs['torch_dtype'] = torch_dtype

    # Add user kwargs
    load_kwargs.update(kwargs)

    try:
        model = model_class.from_pretrained(model_name, **load_kwargs)

        # Verify model is on the right device
        if torch.cuda.is_available() and device_map != "cpu":
            first_param_device = next(model.parameters()).device
            if first_param_device.type == 'cpu':
                print(f"[Compat] Warning: Model loaded on CPU despite CUDA available!")
                print(f"[Compat] Attempting to move model to CUDA...")
                try:
                    model = model.cuda()
                except Exception as e:
                    print(f"[Compat] Could not move to CUDA: {e}")

        return model

    except TypeError as e:
        # If error is about unsupported parameter, retry with minimal params
        if "unexpected keyword argument" in str(e):
            # Retry with minimal parameters
            minimal_kwargs = {k: v for k, v in load_kwargs.items()
                            if k in ['device_map', 'torch_dtype']}
            model = model_class.from_pretrained(model_name, **minimal_kwargs)

            # Try to move to CUDA if available
            if torch.cuda.is_available() and device_map != "cpu":
                try:
                    model = model.cuda()
                except:
                    pass

            return model
        else:
            raise


def load_tokenizer_safe(tokenizer_class, model_name, **kwargs):
    """
    Load tokenizer and ensure it has a pad token.

    Args:
        tokenizer_class: Tokenizer class (e.g., AutoTokenizer)
        model_name: Model name or path
        **kwargs: Additional loading arguments

    Returns:
        Loaded tokenizer with pad_token set
    """
    tokenizer = tokenizer_class.from_pretrained(model_name, **kwargs)
    ensure_pad_token(tokenizer)
    return tokenizer


def generate_safe(model, input_ids=None, inputs_embeds=None, max_new_tokens=50, **kwargs):
    """
    Generate tokens with version-compatible parameters.

    Args:
        model: HuggingFace model
        input_ids: Optional input token IDs
        inputs_embeds: Optional input embeddings
        max_new_tokens: Number of new tokens to generate
        **kwargs: Additional generation arguments

    Returns:
        Generated token IDs
    """
    generate_kwargs = {}

    # Handle inputs
    if input_ids is not None:
        generate_kwargs['input_ids'] = input_ids
    elif inputs_embeds is not None:
        generate_kwargs['inputs_embeds'] = inputs_embeds

    # Handle max_new_tokens vs max_length
    # Try max_new_tokens first (preferred in newer versions)
    if IS_NEW_VERSION or TRANSFORMERS_VERSION >= (4, 27, 0):
        generate_kwargs['max_new_tokens'] = max_new_tokens
    else:
        # Fallback to max_length for very old versions
        if input_ids is not None:
            generate_kwargs['max_length'] = input_ids.shape[1] + max_new_tokens
        else:
            generate_kwargs['max_new_tokens'] = max_new_tokens

    # Add user kwargs
    generate_kwargs.update(kwargs)

    try:
        return model.generate(**generate_kwargs)
    except TypeError as e:
        # If max_new_tokens is not supported, fall back to max_length
        if "max_new_tokens" in str(e) and input_ids is not None:
            generate_kwargs.pop('max_new_tokens', None)
            generate_kwargs['max_length'] = input_ids.shape[1] + max_new_tokens
            return model.generate(**generate_kwargs)
        else:
            raise


def compute_loss_safe(logits, labels, ignore_index=-100):
    """
    Compute cross-entropy loss in a safe way.

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation

    Returns:
        Loss tensor
    """
    # Shift for next-token prediction
    if logits.dim() == 3:  # (batch, seq_len, vocab)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels

    # Reshape
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=ignore_index,
        reduction='mean'
    )

    return loss


def model_forward_safe(model, input_ids=None, inputs_embeds=None, attention_mask=None,
                      labels=None, return_dict=None, **kwargs):
    """
    Perform model forward pass with maximum compatibility.

    Args:
        model: HuggingFace model
        input_ids: Optional input token IDs
        inputs_embeds: Optional input embeddings
        attention_mask: Optional attention mask
        labels: Optional labels for loss computation
        return_dict: Whether to return a dict (None = auto)
        **kwargs: Additional model arguments

    Returns:
        Model outputs
    """
    forward_kwargs = {}

    if input_ids is not None:
        forward_kwargs['input_ids'] = input_ids
    elif inputs_embeds is not None:
        forward_kwargs['inputs_embeds'] = inputs_embeds

    if attention_mask is not None:
        forward_kwargs['attention_mask'] = attention_mask

    if labels is not None:
        forward_kwargs['labels'] = labels

    # Explicitly set return_dict for compatibility
    if return_dict is not None:
        forward_kwargs['return_dict'] = return_dict
    elif IS_OLD_VERSION:
        # Force return_dict=True for consistency in old versions
        forward_kwargs['return_dict'] = True

    forward_kwargs.update(kwargs)

    return model(**forward_kwargs)


# Convenience function to check version
def check_version_compatibility():
    """Print version compatibility information."""
    print(f"Transformers version: {transformers.__version__}")
    print(f"Version tuple: {TRANSFORMERS_VERSION}")
    print(f"Old version (4.x): {IS_OLD_VERSION}")
    print(f"New version (5.x+): {IS_NEW_VERSION}")

    if IS_OLD_VERSION:
        print("\n⚠️  Running with transformers 4.x - using compatibility mode")
    else:
        print("\n✓ Running with transformers 5.x+ - using modern API")


if __name__ == "__main__":
    check_version_compatibility()
