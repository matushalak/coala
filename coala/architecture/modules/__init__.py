from . import ConvNeXt, ConvNet, ViT


MODULE_FAMILIES = {
    "convnet": ConvNet,
    "convnext": ConvNeXt,
    "vit": ViT,
}


def resolve_module_family(module_name_or_impl):
    if not isinstance(module_name_or_impl, str):
        return module_name_or_impl
    family = MODULE_FAMILIES.get(module_name_or_impl.lower())
    assert family is not None
    return family


__all__ = [
    "ConvNet",
    "ConvNeXt",
    "ViT",
    "MODULE_FAMILIES",
    "resolve_module_family",
]
