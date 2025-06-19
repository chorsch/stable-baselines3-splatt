import torch 
import numpy as np
import matplotlib.pyplot as plt
import math
import wandb

def weights_logits_loss_fn(weights_logits):
    if weights_logits.shape[-2] != weights_logits.shape[-1]:
            weights_logits = weights_logits[:,:-1]
    mask = (weights_logits <= 0).to(dtype=torch.float64)
    return ((weights_logits*mask)**2).sum()

def calculate_eta(weights):
    b,l,h,w = weights.shape
    eta = torch.eye(h).repeat(b,1,1).to('cuda')
    for i in range(l):
        eta = torch.bmm(weights[:,i,:,:],eta)
    return eta

def plot_weights(obs, weights, action, name):
    num_plots = 3
    fig, ax = plt.subplots(1,num_plots, figsize=(16, 9))
    
    #plot the observation
    img = torch.permute(obs,(1,2,0)).detach().cpu().numpy()/255.0
    w,h = img.shape[0]//8, img.shape[1]//8
    if weights is not None:
        dx = dy = 64//int(np.sqrt(weights.shape[0]))
        # Custom (rgb) grid color
        grid_color = [0,0,1]
        # Modify the image to include the grid
        img[:,::dy,:] = grid_color
        img[::dx,:,:] = grid_color
    ax[0].imshow(img)

    ax[0].set_axis_off()
    fig.suptitle(f'{name}, action:{action.item()}')
    if weights is not None:
        im = ax[2].imshow(weights.detach().cpu().numpy().T, cmap="viridis")
        d = int(np.sqrt(weights.shape[0]))
        im2 = ax[1].imshow(weights.transpose(-2, -1).sum(-1).reshape(w,h).detach().cpu().numpy()/d**2, cmap="viridis")
        ax[1].set_axis_off()
        fig.colorbar(im2, ax=ax[1], orientation="horizontal")
        fig.colorbar(im, ax=ax[2], orientation="horizontal")
        wandb.log({f"weights heatmap {name}": plt})
    plt.close()

def get_entropy(weights):
    weights = weights.detach()
    b,h,w = weights.shape
    weights = weights.flatten(-2, -1)
    #assert weights.sum()/(h) == b, f'{weights.sum()/(h)}, {b}'
    assert weights.shape == (b,h*w)
    #weights = torch.nn.functional.softmax(weights, dim=-1)
    assert weights.shape == (b,h*w)
    log_weights = torch.log2(weights+1e-8)
    assert (-1.0 * weights * log_weights).shape == (b,h*w)
    out = (-1.0 * weights * log_weights).sum(dim = -1)
    assert out.shape == (b,), f'{b}, {out.shape}'
    return out.mean()/(np.log2(h*w)*h)

def calculate_weights_coef_sin(t, t_max, c_max):
    return (c_max/2) * np.sin(t*np.pi/t_max - np.pi/2) + (c_max/2)

def calculate_weights_coef_linear(t, t_max, c_max):
    return (c_max/t_max) * t

def bottom_k_inputs(k, weights, add_zero_attn=False):
    b, d, _ = weights.shape #entity_dim
    p = weights.transpose(-2, -1).sum(-1)
    assert p.shape == (b, d)
    bot_k = d - k
    return ((torch.topk(p, bot_k, dim=-1, largest=False)[0].sum(dim=-1)**2)).mean() * (bot_k/d)
    # return ((torch.topk(p, bot_k, dim=-1, largest=False)[0].sum(dim=-1)**2)*torch.nn.functional.relu(adv)).mean() * (bot_k/d)

def L1_mask(eta, add_zero_attn=False):
    return eta.sum()

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe