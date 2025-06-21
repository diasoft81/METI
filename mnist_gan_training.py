import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
noise_dim = 100
embedding_dim = 10
hidden_dim = 256
batch_size = 128
epochs = 30
lr = 2e-4

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
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

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(28*28 + embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        x = torch.cat([images, self.label_emb(labels)], dim=1)
        out = self.net(x)
        return out

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(noise_dim, embedding_dim).to(device)
    D = Discriminator(embedding_dim).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(10, noise_dim, device=device)
    fixed_labels = torch.arange(10, device=device)

    for epoch in range(1, epochs + 1):
        for real_imgs, real_labels in dataloader:
            batch_size_curr = real_imgs.size(0)
            real_imgs = real_imgs.view(batch_size_curr, -1).to(device)
            real_labels = real_labels.to(device)

            # Train Discriminator
            z = torch.randn(batch_size_curr, noise_dim, device=device)
            fake_labels = torch.randint(0, 10, (batch_size_curr,), device=device)
            fake_imgs = G(z, fake_labels)

            real_targets = torch.ones(batch_size_curr, 1, device=device)
            fake_targets = torch.zeros(batch_size_curr, 1, device=device)

            D_real = D(real_imgs, real_labels)
            D_fake = D(fake_imgs.detach(), fake_labels)

            loss_real = criterion(D_real, real_targets)
            loss_fake = criterion(D_fake, fake_targets)
            loss_D = (loss_real + loss_fake) / 2

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # Train Generator
            D_fake_for_G = D(fake_imgs, fake_labels)
            loss_G = criterion(D_fake_for_G, real_targets)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        print(f"Epoch [{epoch}/{epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

    torch.save(G.state_dict(), 'mnist_cgan_generator.pth')

if __name__ == '__main__':
    train()
