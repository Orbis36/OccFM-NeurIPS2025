import torch
import torch.nn as nn

class VaeQuant(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mus, self.logvars = torch.zeros(1), torch.zeros(1)

        self.mu_sigma_cache = False
        self.latent_cache = False
        self.skip = kwargs.get('skip', False)

    def forward(self, data_dict):

        if self.skip:
            return data_dict

        compressed_features = data_dict['compressed_features']

        dim = compressed_features.shape[1] // 2
        mu = compressed_features[:, :dim]
        sigma = torch.exp(compressed_features[:, dim:] / 2).to(compressed_features.dtype)
        eps = torch.randn_like(mu)

        sampled_features = mu + sigma * eps
        data_dict['sampled_features'] = sampled_features

        self.mus, self.logvars = mu, sigma

        data_dict['mu'], data_dict['sigma'] = (mu, sigma) if self.mu_sigma_cache else (None, None)
        return data_dict

    def get_loss(self):

        loss = torch.sum(-0.5 * torch.sum(1 + self.logvars - self.mus.pow(2) - self.logvars.exp()))
        pixels = torch.sum(torch.prod(torch.as_tensor(self.mus.shape)))
        return loss / pixels
