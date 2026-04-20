import torch


def load_with_mismatch(model, pretrained_state_dict):
    def repeat(pretrained_param, model_param):
        if pretrained_param.shape == model_param.shape:
            return pretrained_param

        ns = model_param.shape
        expanded = pretrained_param
        for dim in range(len(ns)):
            if pretrained_param.shape[dim] < ns[dim]:
                repeats = ns[dim] // pretrained_param.shape[dim]
                expanded = expanded.repeat_interleave(repeats, dim=dim)
            if pretrained_param.shape[dim] > ns[dim]:
                expanded = torch.narrow(pretrained_param, dim, 0, ns[dim])
        return expanded

    model_state_dict = model.state_dict()

    for name, pretrained_param in pretrained_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = repeat(
                pretrained_param.clone(),
                model_state_dict[name],
            )

    model.load_state_dict(model_state_dict)
    return model


def load_with_mismatch_from_weights(model, weights, progress: bool = True):
    pretrained_state_dict = weights.get_state_dict(progress=progress)
    return load_with_mismatch(model, pretrained_state_dict)
