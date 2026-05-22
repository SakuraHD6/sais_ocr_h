"""
EfficientNet/Transformer + ArcFace classifier for ancient character recognition.
Backbone architecture is determined from the checkpoint at runtime.
"""
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        features = nn.functional.normalize(features)
        weight = nn.functional.normalize(self.weight)
        cos_theta = torch.mm(features, weight.t()).clamp(-1, 1)
        return cos_theta * self.s


class CharacterClassifier:
    def __init__(self, weights_path, device="cuda:0", id_to_char_path=None):
        self.device = torch.device(device)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        num_classes = checkpoint["num_classes"]
        backbone_name = checkpoint.get("backbone", "efficientnet_b0")
        img_size = checkpoint.get("img_size", 128)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Timm model name mapping
        TIMM_NAMES = {
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet_b2": "efficientnet_b2",
            "efficientnet_b3": "efficientnet_b3",
            "efficientnetv2_s": "efficientnetv2_s",
            "swin_tiny": "swin_tiny_patch4_window7_224",
        }
        timm_name = TIMM_NAMES.get(backbone_name, "efficientnet_b0")

        # Load ID → Chinese character mapping
        self.id_to_char = {}
        if id_to_char_path:
            with open(id_to_char_path) as f:
                self.id_to_char = json.load(f)

        # Build backbone
        backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
        in_features = backbone.num_features

        # Build ArcFace head
        self.arcface = ArcFaceHead(in_features, num_classes)

        self.backbone = backbone.to(self.device)
        self.arcface = self.arcface.to(self.device)

        # Load weights
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.arcface.load_state_dict(checkpoint["arcface_state_dict"])

        self.backbone.eval()
        self.arcface.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def predict(self, pil_image):
        """Classify a single character crop. Returns (char, confidence)."""
        img = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(img)
            logits = self.arcface(features)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(1)

        class_id = self.idx_to_class[pred_idx.item()]
        # Convert numeric ID to Chinese character if mapping exists
        char = self.id_to_char.get(class_id)
        if char is None and '_' in class_id:
            # Merged class (e.g. "0011_0012_0013"): take the first mapped component
            for cid in class_id.split('_'):
                c = self.id_to_char.get(cid)
                if c:
                    char = c
                    break
        if char is None:
            char = class_id
        return char, confidence.item()
