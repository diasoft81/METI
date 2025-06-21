import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Hyperparameters should match training
noise_dim = 100
embedding_dim = 10
hidden_dim = 256

class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        input_dim = noise_dim + embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 28*28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_emb(labels)], dim=1)
        out = self.net(x)
        return out

@st.cache_resource
def load_generator(path='mnist_cgan_generator.pth'):
    device = torch.device('cpu')
    G = Generator(noise_dim, embedding_dim).to(device)
    G.load_state_dict(torch.load(path, map_location=device))
    G.eval()
    return G

def generate_images(generator, digit, num_images=5):
    device = next(generator.parameters()).device
    noise = torch.randn(num_images, noise_dim, device=device)
    labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
    with torch.no_grad():
        imgs = generator(noise, labels)
    imgs = imgs.view(num_images, 28, 28)
    imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
    imgs = imgs.mul(255).clamp(0, 255).byte().cpu().numpy()
    return imgs

st.title("MNIST Handwritten Digit Generator")

digit = st.selectbox("Select digit", list(range(10)), index=0)
if st.button("Generate 5 Images"):
    G = load_generator()
    images = generate_images(G, int(digit), num_images=5)
    cols = st.columns(5)
    for col, img in zip(cols, images):
        col.image(Image.fromarray(img, mode='L'), caption=f"Digit {digit}")
