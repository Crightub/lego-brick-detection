import timm
import torch.nn as nn


class ViTClassifier(nn.Module):
    """
    ViT-S/16 backbone with a separate classification head.
    The backbone outputs embeddings fed into ArcFaceLoss during training.
    The classification head is used at inference time.
    """
    def __init__(self, num_classes: int, embedding_dim: int = 384):
        super().__init__()

        # Load ViT-S/16 with pretrained ImageNet weights, no classification head
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=0,       # removes the default head — outputs raw embeddings
            img_size=128         # match your crop size
        )

        self.embedding_dim = embedding_dim  # ViT-S outputs 384-dim embeddings

        # Classification head used at inference — also trained jointly
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embeddings = self.backbone(x)         # (B, 384)
        logits = self.classifier(embeddings)  # (B, num_classes)
        return embeddings, logits