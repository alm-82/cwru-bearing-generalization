"""EfficientNet-B0 architecture, image transforms, and historical stage modes."""
import torch
from torch import nn
from .data import validate_config
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(config, pretrained=True):
    """Build the historical classifier; seeded profiles reset the global torch RNG.

    pretrained=True may download torchvision DEFAULT ImageNet weights.
    """
    validate_config(config)
    # Cross-load resets torch's RNG; cross-fault-size retains the ambient state.
    if config["seed"] is not None:
        torch.manual_seed(config["seed"])
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(config["dropout"]),
                                    nn.Linear(model.classifier[1].in_features, len(config["classes"])))
    configure_stage(model, 1)
    return model


def configure_stage(model, stage):
    if stage not in {1, 2}:
        raise ValueError("Stage must be 1 or 2")
    for p in model.features.parameters():
        p.requires_grad = False
    if stage == 2:
        for p in model.features[-1].parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_training_mode(model, mode):
    model.train()
    if mode == "features_eval":
        model.features.eval()
    elif mode == "last_projection_train":
        for block in model.features[:-1]:
            block.eval()
        model.features[-1].train()
    elif mode == "all_bn_eval":
        # Frozen earlier stochastic-depth layers remain in training mode.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_loaders(image_root, config):
    """Load three class-indexed image splits using the historical transform order."""
    validate_config(config)
    from pathlib import Path
    resize = transforms.Resize((config["image_size"], config["image_size"]), antialias=True)
    gray = transforms.Grayscale(num_output_channels=3)
    first = [gray, resize] if config["transform_order"] == "grayscale_then_resize" else [resize, gray]
    transform = transforms.Compose(first + [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    result = {}
    mapping = {str(i): i for i in range(len(config["classes"]))}
    for split in ["train", "val", "test"]:
        ds = datasets.ImageFolder(Path(image_root) / split, transform=transform)
        if ds.class_to_idx != mapping:
            raise ValueError(f"ImageFolder class mapping differs in {split}: {ds.class_to_idx}")
        generator = None
        if split == "train" and config["loader_generator_seed"] is not None:
            generator = torch.Generator().manual_seed(config["loader_generator_seed"])
        result[split] = torch.utils.data.DataLoader(ds, batch_size=config["batch_size"],
            shuffle=split == "train", num_workers=config["num_workers"],
            pin_memory=config["pin_memory"], generator=generator)
    return result
