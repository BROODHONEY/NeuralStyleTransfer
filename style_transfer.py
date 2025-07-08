import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
from PIL import Image
import numpy as np

weights = VGG19_Weights.DEFAULT
model = models.vgg19(weights=weights).features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_layers = ['0', '5', '10', '19']
        self.model = models.vgg19(weights=weights).features[:21].eval().to(device)  # Extract the first 21 layers of VGG19
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.chosen_layers:
                features[name] = x
        return features

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t()) / (c * h * w)
    return gram

def get_loss(gen_feats, content_feats, style_feats):
    content_loss = torch.nn.functional.mse_loss(gen_feats['10'], content_feats['10'])
    style_loss = 0.0
    for layer in ['0', '5', '10', '19']:
        gm_gen = gram_matrix(gen_feats[layer])
        gm_style = gram_matrix(style_feats[layer])
        style_loss += torch.nn.functional.mse_loss(gm_gen, gm_style)
    return content_loss, style_loss

def load_image(image_input):
    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    image = transform(image).unsqueeze(0)
    return image.to(device)

def run_style_transfer(content_img, style_img, steps=3000, style_weight=1000, content_weight=1, learning_rate=0.003):
    model = VGG().to(device).eval()
    with torch.no_grad():
        content_feats = model(content_img)
        style_feats = model(style_img)

    generated_img = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_img], lr=learning_rate)

    for i in range(steps):
        optimizer.zero_grad()
        gen_feats = model(generated_img)
        content_loss, style_loss = get_loss(gen_feats, content_feats, style_feats)
        total_loss = content_weight * content_loss + style_weight * style_loss
  
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step [{i}/{steps}], total_loss: {total_loss.item():.4f}")

    return generated_img

def save_output(tensor, path="output.png"):
    image = tensor.clone().detach().squeeze()
    image = image.clamp(0, 1)
    save_image(image, path)
    return path