import torch


def _register_param_isolation(server, is_private):
    """Keep parameters matching is_private(key) client-private: drop them from the
    aggregated update (so the global model is not touched) and from the distributed
    state dict (so clients keep their own copy via load_state_dict(strict=False))."""

    def _strip_update(server):
        for key in [k for k in server.update.keys() if is_private(k)]:
            server.update.pop(key)

    def _strip_distribute(server):
        for key in [k for k in server.distribute_dict.keys() if is_private(k)]:
            server.distribute_dict.pop(key)

    server.register_func(_strip_update, "before_update_global")
    server.register_func(_strip_distribute, "before_distribute_global")


# head = final classifier only (ResNet: linear), matching the canonical
# FedRep/FedPer setup. Override head_keys for other backbones (e.g. MobileNet:
# "classifier", DenseNet: "fc").
HEAD_KEYS = ("linear",)


def _is_head(key, head_keys=HEAD_KEYS):
    return any(h in key for h in head_keys)


def use_fedbn(server):
    _register_param_isolation(server, lambda k: "bn" in k or "shortcut.1" in k)


def use_fedper(server, head_keys=HEAD_KEYS):
    _register_param_isolation(server, lambda k: _is_head(k, head_keys))


def _set_trainable(model, train_head, head_keys=HEAD_KEYS):
    for name, p in model.named_parameters():
        p.requires_grad = _is_head(name, head_keys) if train_head else (not _is_head(name, head_keys))


def use_fedrep(clients, server, local_steps, head_steps=None, head_keys=HEAD_KEYS):
    use_fedper(server, head_keys)  # head stays private, base is aggregated
    # default: train head for the first n-1 local steps, then base for the last step
    head_steps = local_steps - 1 if head_steps is None else head_steps

    def _begin(client):
        client.training_info["fedrep_step"] = 0
        _set_trainable(client.local_model, train_head=True, head_keys=head_keys)  # phase 1: freeze base, train head

    def _after_step(client):
        client.training_info["fedrep_step"] += 1
        if client.training_info["fedrep_step"] == head_steps:
            _set_trainable(client.local_model, train_head=False, head_keys=head_keys)  # phase 2: freeze head, train base

    for client in clients:
        client.register_func(_begin, "before_local_training")
        client.register_func(_after_step, "after_local_update")
