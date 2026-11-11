"""Analytical FLOP counts (multiply-add = 2 FLOPs), in the style of Kaplan et al. 2020.

These are estimates used to compare pipelines at matched compute, not profiler measurements.
"""


def lstm_lm_fwd_flops_per_token(vocab, emb, hid):
    lstm = 2 * 4 * hid * (emb + hid)
    proj = 2 * hid * emb
    logits = 2 * emb * vocab
    return lstm + proj + logits


def gpt_param_counts(n_layer, d_model, ctx, vocab):
    non_embedding = 12 * n_layer * d_model * d_model
    embedding = vocab * d_model + ctx * d_model     # tied token embedding + learned positions
    return {"non_embedding": non_embedding, "embedding": embedding, "total": non_embedding + embedding}


def gpt_fwd_flops_per_token(n_layer, d_model, ctx, vocab):
    return 2 * 12 * n_layer * d_model * d_model + 2 * n_layer * ctx * d_model + 2 * vocab * d_model


def gpt_train_flops_per_token(n_layer, d_model, ctx, vocab):
    return 3 * gpt_fwd_flops_per_token(n_layer, d_model, ctx, vocab)
