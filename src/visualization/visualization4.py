import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

from src.models import VAE, VLAE, FullCovVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

DATA_ROOT = "datasets/MNIST"   

dataset = MNIST(
    root=DATA_ROOT,
    train=True,
    download=False,     
    transform=transforms.ToTensor()
)

x, _ = next(iter(dataset))
x = x.to(device)   

vae = VAE(
    z_dim=50,
    output_dist="gaussian",
    logit_transform=False,
    dataset="MNIST",
    x_dim=(1, 28, 28),
    enc_dim=500,
    dec_dim=500).to(device)
vlae = VLAE(
    z_dim=50,
    output_dist="gaussian",
    logit_transform=False,
    dataset="MNIST",
    n_update=1,
    update_lr=0.5,
    x_dim=(1, 28, 28),
    enc_dim=500,
    dec_dim=500).to(device)
full_vae = FullCovVAE(
    dataset="MNIST",
    z_dim=50,
    output_dist="gaussian",
    x_dim=(1, 28, 28),
    enc_dim=500,
    dec_dim=500).to(device)

vae.load_state_dict(torch.load("checkpoints/MNIST/VAE/2025-12-23_10:22:56.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0/1063.pkl", map_location=device))
vlae.load_state_dict(torch.load("checkpoints/MNIST/VLAE/2025-12-20_16:08:01.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0/1416.pkl", map_location=device))
full_vae.load_state_dict(torch.load("checkpoints/MNIST/FullCovVAE/2025-12-22_22:34:12.logit_transform=False.out_dist=gaussian.z_dim=50.hid_dim=500.lr=0.0005.weight_decay=0.0/1373.pkl", map_location=device))

vae.eval()
vlae.eval()
full_vae.eval()

with torch.no_grad():
    q_z_x, _ = vae.encoder(x.view(x.size(0), -1))
    logvar = q_z_x.logvar[0]
    cov_vae = torch.diag(torch.exp(logvar)).cpu().numpy()


with torch.no_grad():
    mu, logvar, L_params= full_vae.encoder(x.view(x.size(0), -1))
    L = full_vae.build_cholesky(logvar, L_params)
    cov = (L @ L.transpose(-1,-2))[0]
cov_full = cov.cpu().numpy()


with torch.no_grad():
    q_z_x, _ = vlae.encoder(x.view(x.size(0), -1))
    mu = q_z_x.mu

    for i in range(vlae.n_update):
        p_x_z, W_dec = vlae.decoder(mu, compute_jacobian=True)
        mu_new, precision = vlae.solve_mu(x, mu, p_x_z, W_dec)
        lr = vlae.update_rate(i)
        mu = (1 - lr) * mu + lr * mu_new

    p_x_z, W_dec = vlae.decoder(mu, compute_jacobian=True)
    var_inv = torch.exp(-vlae.decoder.logvar).unsqueeze(1)
    precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
    precision += torch.eye(precision.shape[1]).to(device)

    cov_vlae = torch.inverse(precision[0]).cpu().numpy()

out_dir = "out_images"
plt.figure(figsize=(4,4))
# plt.subplot(1, 3, 1)
plt.imshow(np.log(cov_vae + 1e-8), cmap="gray")
plt.axis("off")
plt.savefig(f"{out_dir}/posterior_covariance1.png", dpi=150)

# plt.subplot(1, 3, 2)
cov_s=np.clip(cov_full, 1e-12, None)
plt.imshow(np.log(cov_s), cmap="gray")
plt.axis("off")
plt.savefig(f"{out_dir}/posterior_covariance2.png", dpi=150)

# plt.subplot(1, 3, 3)
plt.imshow(cov_vlae, cmap="gray")
plt.axis("off")
plt.savefig(f"{out_dir}/posterior_covariance3.png", dpi=150)
