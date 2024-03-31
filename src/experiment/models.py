import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms
import geomloss
from geomloss import SamplesLoss
import numpy as np
import random
import math

def torch_cov(input_vec:torch.tensor):
    x = input_vec- torch.mean(input_vec,axis=0)
    cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
    return cov_matrix

def binary_classification_loss_g(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    t_tpred = (concat_pred[:, 3] + 0.001) / 1.002
    r_t1 = torch.sum(t_true) * 1.0 / t_true.shape[0] + 0.00001
    r_t0 = 1 - r_t1
    print(torch.cat((t_true.reshape(-1,1),t_pred.reshape(-1,1),t_tpred.reshape(-1,1)), 1))
    loss  =  torch.sum(-(r_t0) * t_true * torch.log(t_pred) - (r_t1) *(1 - t_true) * torch.log(1 - t_pred))
    loss1 = torch.sum(-(r_t0) * t_true * torch.log(1- t_tpred))
    return loss + loss1

def binary_classification_loss_y(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = (concat_pred[:, 0]  + 0.001 ) / 1.002
    y1_pred = (concat_pred[:, 1]  + 0.001 ) / 1.002
    r_t1 = torch.sum(t_true) * 1.0 / t_true.shape[0] + 0.00001
    r_t0 = 1 - r_t1
    loss0 = torch.sum((1 - t_true) * (- y_true * torch.log(y0_pred) - (1 - y_true) * torch.log(1 - y0_pred)))
    loss1 = torch.sum(t_true * (- y_true * torch.log(y1_pred) - (1 - y_true) * torch.log(1 - y1_pred)))
    return loss0 + loss1

def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    r_t1 = torch.sum(t_true) * 1.0 / t_true.shape[0] + 0.00001
    r_t0 = 1 - r_t1
    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss = loss0 + loss1
    return  loss

def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 1]
    return torch.sum(F.binary_cross_entropy_with_logits(t_true, t_pred))

def SDEL(concat_true, concat_pred, alpha = 1):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    t0_x = concat_pred[:, 2:202]
    t1_x = concat_pred[:, 202:402]
    t0_x_cov = torch_cov(t0_x)
    t1_x_cov = torch_cov(t1_x)
    loss0 = 2 * (t0_x_cov.trace())
    loss1 = 2 * (t1_x_cov.trace())
    t0_x_mean = torch.mean(t0_x, 0)
    t1_x_mean = torch.mean(t1_x, 0)
    loss =  torch.sum(torch.square(t0_x_mean - t1_x_mean)) + 0.00001
    # print("loss0: {}, loss1: {}, loss: {}, Triplet Loss:{}".format(loss0,loss1,loss,alpha * (loss0 + loss1) / loss))
    return alpha * ((loss0 + loss1) / loss)

def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)

def make_orthogonal_reg_loss(concat_true, concat_pred, fn, global_input, ratio=0.01):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    t_pred = concat_pred[:, 2]
    epsilons = concat_pred[:, 3]
    t_pred = (t_pred + 0.001) / 1.002
    # Perturbation
    y0_pert = y0_pred + epsilons * (t_true - t_pred)
    # Treatment effect proxy with average treatment effect of current model fit
    global_pred = fn(torch.tensor(global_input, dtype=torch.float32).to('cuda'))
    global_pred = global_pred.cpu().detach().numpy()
    y0_pred_global = global_pred[:, 0].copy()
    y1_pred_global = global_pred[:, 1].copy()
    psi = (y1_pred_global - y0_pred_global).mean()
    # Orthogonal regularization
    orthogonal_regularization = torch.sum(torch.square(y_true - t_true * psi - y0_pert))
    # final
    loss = ratio * orthogonal_regularization
    return loss

def wasserstein_loss(concat_true, concat_pred, ratio=0.01):
    """ Returns the Wasserstein distance between treatment groups """
    t_true = torch.tensor(concat_true[:, 1].clone().detach(), dtype=torch.bool)
    y1_repre = concat_pred[:, -200:][t_true]
    y0_repre = concat_pred[:, -400:-200][~t_true]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(y1_repre, y0_repre)
    return ratio * imbalance_loss

def mmd2_loss(concat_true, concat_pred, ratio=0.01):
    t_true = torch.tensor(concat_true[:, 1].clone().detach(), dtype=torch.bool)
    y1_repre = concat_pred[:, -200:][t_true]
    y0_repre = concat_pred[:, -400:-200][~t_true]
    samples_loss = SamplesLoss(loss="energy", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(y1_repre, y0_repre)
    return ratio * imbalance_loss

def l2loss(net, mylambda=0.01):
    l2_reg_loss = 0.0
    for param in net.parameters():
        l2_reg_loss += torch.norm(param, p=2)
    return  mylambda * l2_reg_loss

def TTELoss(concat_true, concat_pred, fn, global_input, ratio=0.01):
    return regression_loss(concat_true, concat_pred) + SDEL(concat_true, concat_pred, alpha=10) + l2loss(fn,mylambda=0.01)


def make_tarreg_loss(ratio=1., dragonnet_loss=TTELoss):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred,  fn, global_input, ratio=1):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred, fn, global_input, ratio)
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.001) / 1.002
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))
        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss

def weights_init_normal(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.normal_(params.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(params.bias)

def weights_init_uniform(params):
    if isinstance(params, nn.Linear):
        limit = math.sqrt(6 / (params.weight[1] + params.weight[0]))
        torch.nn.init.uniform_(params.weight, a=-limit, b=limit)
        torch.nn.init.zeros_(params.bias)

class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()
        # building epsilon trainable weight
        self.weights = nn.Parameter(torch.Tensor(1, 1))
        # initializing weight parameter with RandomNormal
        nn.init.normal_(self.weights, mean=0, std=0.05)
    def forward(self, inputs):
        return torch.mm(torch.ones_like(inputs)[:, 0:1], self.weights.T)

class GluLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.glu = nn.GLU()
        self.bn = nn.BatchNorm1d(out_features, momentum=0.01)
    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        a = self.bn(a)
        b = self.bn(b)
        return self.glu(torch.cat((a, b), dim=1))

class FE_CFR(nn.Module):
    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(FE_CFR, self).__init__()
        self.representation_block = nn.Sequential(
            GluLayer(in_features=in_features, out_features=out_features[0]),
            nn.GELU(),
            GluLayer(in_features=out_features[0], out_features=out_features[0]),
            nn.GELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.GELU(),
        )
        # -----------Propensity Head
        # self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]),
        #                                    nn.Sigmoid())
        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[0]),
                                     nn.GELU())
        self.t0_out =  nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.GELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )
        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[0]),
                                     nn.GELU())
        self.t1_out = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.GELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )
        # self.init_params()

    def init_params(self, std=1):
        self.representation_block.apply(weights_init_normal)
        # self.t_predictions.apply(weights_init_normal)
        self.t0_head.apply(weights_init_normal)
        self.t1_head.apply(weights_init_normal)
        self.t0_out.apply(weights_init_normal)
        self.t1_out.apply(weights_init_normal)

    def forward(self, x):
        x = self.representation_block(x)
        # ------propensity scores
        # propensity_head = self.t_predictions(x)
        t0_head = self.t0_head(x)
        t0_out = self.t0_out(t0_head)
        t1_head = self.t1_head(x)
        t1_out = self.t1_out(t1_head)
        return torch.cat((t0_out, t1_out, t0_head, t1_head, x), 1)

class donut(nn.Module):
    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(donut, self).__init__()
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )
        # -----------Propensity Head
        self.t_predictions = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features[2]),
                                           nn.Sigmoid())
        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU())
        self.t0_out = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                    nn.ELU(),
                                    nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                    )
        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU())
        self.t1_out = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()

    def init_params(self, std=1):
        self.representation_block.apply(weights_init_normal)
        self.t_predictions.apply(weights_init_normal)
        self.t0_head.apply(weights_init_normal)
        self.t1_head.apply(weights_init_normal)
        self.t0_out.apply(weights_init_normal)
        self.t1_out.apply(weights_init_normal)

    def forward(self, x):
        rep_block = self.representation_block(x)
        propensity_head = self.t_predictions(x)
        epsilons = self.epsilon(propensity_head)
        t0_head = self.t0_head(rep_block)
        t0_out = self.t0_out(t0_head)
        t1_head = self.t1_head(rep_block)
        t1_out = self.t1_out(t1_head)
        return torch.cat((t0_out, t1_out, propensity_head, epsilons, t0_head, t1_head, rep_block), 1)

class DragonNet(nn.Module):
    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(DragonNet, self).__init__()
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU()
        )
        # -----------Propensity Head
        self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]),
                                           nn.Sigmoid())

        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU())
        self.t0_out = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                    nn.ELU(),
                                    nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                    )
        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU())
        self.t1_out = nn.Sequential(nn.Linear(in_features=out_features[1], out_features=out_features[1]),
                                     nn.ELU(),
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2])
                                     )

        self.epsilon = EpsilonLayer()

    def init_params(self, std=1):
        self.representation_block.apply(weights_init_normal)
        self.t_predictions.apply(weights_init_normal)
        self.t0_head.apply(weights_init_normal)
        self.t1_head.apply(weights_init_normal)
        self.t0_out.apply(weights_init_normal)
        self.t1_out.apply(weights_init_normal)

    def forward(self, x):
        rep_block = self.representation_block(x)
        propensity_head = self.t_predictions(rep_block)
        epsilons = self.epsilon(propensity_head)
        t0_head = self.t0_head(rep_block)
        t0_out = self.t0_out(t0_head)
        t1_head = self.t1_head(rep_block)
        t1_out = self.t1_out(t1_head)
        return torch.cat((t0_out, t1_out, propensity_head, epsilons,  t0_head, t1_head, rep_block), 1)

class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        self.val_loss_min = val_loss