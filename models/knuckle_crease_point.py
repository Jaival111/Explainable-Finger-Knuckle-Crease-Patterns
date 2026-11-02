"""Step 1 - KnuckleCreasePoint (Sec. 3.1).

SuperPoint-style fully-convolutional network that jointly detects knuckle
crease keypoints K (locations), their scores S and 256-d descriptors F.
Layer names match the official SuperPoint release so `superpoint_v1.pth`
can be loaded directly as an initialisation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def simple_nms(scores, radius):
    """Fast max-pool based non-maximum suppression on (B, H, W) score maps."""
    def max_pool(x):
        return F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def sample_descriptors(keypoints, descriptors, s=8):
    """Bilinear interpolation of the coarse descriptor map at keypoint locations.

    keypoints: (B, N, 2) xy pixels, descriptors: (B, D, H/8, W/8) -> (B, N, D), L2-normalised.
    """
    b, c, h, w = descriptors.shape
    kp = keypoints - s / 2 + 0.5
    kp = kp / kp.new_tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)])
    kp = kp * 2 - 1
    desc = F.grid_sample(descriptors, kp.view(b, 1, -1, 2), mode="bilinear", align_corners=True)
    desc = F.normalize(desc.reshape(b, c, -1), p=2, dim=1)
    return desc.transpose(1, 2)


class KnuckleCreasePoint(nn.Module):
    def __init__(self, descriptor_dim=256, nms_radius=3, max_keypoints=256, remove_borders=4):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.remove_borders = remove_borders

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        # Shared VGG-style encoder
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        # Interest point decoder (64 cell positions + 1 "no keypoint" dustbin)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        # Descriptor decoder
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, descriptor_dim, 1, 1, 0)

    def forward(self, image):
        """image: (B, 1, H, W) in [0, 1] -> detector logits (B, 65, H/8, W/8), raw descriptors."""
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        logits = self.convPb(self.relu(self.convPa(x)))
        desc = self.convDb(self.relu(self.convDa(x)))
        return logits, desc

    @staticmethod
    def logits_to_heatmap(logits):
        b, _, h, w = logits.shape
        prob = F.softmax(logits, 1)[:, :-1]
        prob = prob.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        return prob.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)

    def extract(self, image, max_keypoints=None):
        """Detect exactly N keypoints per image, sorted by detection score.

        Returns dict with keypoints (B, N, 2) xy, scores (B, N), descriptors (B, N, D).
        """
        n = max_keypoints or self.max_keypoints
        logits, desc = self.forward(image)
        heat = self.logits_to_heatmap(logits)
        nms = simple_nms(heat, self.nms_radius)
        # NMS maxima first; if an image has fewer than N maxima, fall back to the
        # strongest remaining responses so every template has a fixed size.
        sel = torch.where(nms > 0, nms + 1.0, heat * 1e-3)
        b = self.remove_borders
        if b > 0:
            sel[:, :b] = -1
            sel[:, -b:] = -1
            sel[:, :, :b] = -1
            sel[:, :, -b:] = -1
        _, idx = sel.flatten(1).topk(n, dim=1)
        w = heat.shape[-1]
        kpts = torch.stack([idx % w, torch.div(idx, w, rounding_mode="floor")], -1).float()
        scores = heat.flatten(1).gather(1, idx)
        descriptors = sample_descriptors(kpts, desc, 8)
        return {"keypoints": kpts, "scores": scores, "descriptors": descriptors}

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state)
