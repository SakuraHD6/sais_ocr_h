"""
EfficientNet/Transformer + ArcFace classifier for ancient character recognition.
Backbone architecture is determined from the checkpoint at runtime.
"""
import json
import os
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
        self.merged_class_policy = os.getenv("MERGED_CLASS_POLICY", "common").lower()

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
            with open(id_to_char_path, encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict):
                if "id_to_char" in mapping:
                    self.id_to_char = mapping["id_to_char"]
                elif "char_mapping" in mapping:
                    self.id_to_char = mapping["char_mapping"]
                elif "class_to_idx" not in mapping:
                    self.id_to_char = mapping

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

    @staticmethod
    def _cjk_priority(text):
        if len(text) != 1:
            return (4, len(text), text)
        codepoint = ord(text)
        if 0x4E00 <= codepoint <= 0x9FFF:
            return (0, codepoint)
        if 0x3400 <= codepoint <= 0x4DBF:
            return (1, codepoint)
        if 0x20000 <= codepoint <= 0x2EBEF:
            return (2, codepoint)
        return (3, codepoint)

    def _resolve_merged_class(self, class_id):
        candidates = [
            self.id_to_char[cid]
            for cid in class_id.split("_")
            if cid in self.id_to_char
        ]
        if not candidates:
            return None
        if self.merged_class_policy == "first":
            return candidates[0]
        if self.merged_class_policy == "last":
            return candidates[-1]
        return min(candidates, key=self._cjk_priority)

    @staticmethod
    def _is_non_output_label(class_id):
        if not class_id:
            return True
        upper = class_id.upper()
        if upper in {"NONE", "NULL", "UNKNOWN"}:
            return True
        if upper.startswith(("ZH-", "ZHFD-")):
            return True
        if class_id.isdigit():
            return True
        if "_" in class_id and all(part.isdigit() for part in class_id.split("_")):
            return True
        return False

    def _class_id_to_text(self, class_id):
        char = self.id_to_char.get(class_id)
        if char is None and "_" in class_id:
            char = self._resolve_merged_class(class_id)
        if char is not None:
            return char
        if self._is_non_output_label(class_id):
            return None
        return class_id

    def predict(self, pil_image):
        """Classify a single character crop. Returns (text, confidence)."""
        img = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.backbone(img)
            logits = self.arcface(features)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(1)

        class_id = self.idx_to_class[pred_idx.item()]
        return self._class_id_to_text(class_id), confidence.item()
