# from experiment.models import *
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from experiment.models import *
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random

def _split_output(yt_hat, t, y, y_scaler, x, mu_0, mu_1, index, is_targeted_regularization):
    yt_hat = yt_hat.detach().cpu().numpy()
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].copy()
    if is_targeted_regularization:
        eps = yt_hat[:, 3]
    else:
        eps = np.zeros_like(yt_hat[:, 2])
    y = y_scaler.inverse_transform(y.copy())
    var = "{}: average propensity for treated: {} and untreated: {}".format(index, g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)
    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'mu_0': mu_0, 'mu_1': mu_1, 'index': index, 'eps': eps}


def train(train_loader, net, optimizer, criterion, x_train):
    avg_loss = 0
    # iterate through batches
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(labels, outputs, net, x_train)
        loss.backward()
        optimizer.step()
        # keep track of loss and accuracy
        avg_loss += loss
    return avg_loss / len(train_loader)

def valid(valid_loader, net, criterion, x_train):
    avg_loss = 0
    net.eval()
    # iterate through batches
    for i, data in enumerate(valid_loader):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(labels, outputs, net, x_train)
        avg_loss += loss
    return avg_loss / len(valid_loader)



def train_and_predict_dragons(t_tr, y_tr, x_tr, mu_0_tr, mu_1_tr, t_te, y_te, x_te, mu_0_te, mu_1_te, output_dir='',
                              knob_loss=TTELoss, ratio=1., dragon='', val_split=0.2, batch_size=256, idx=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # print(f"batch_size: {batch_size}")
    t_tr = t_tr.reshape(-1, 1)
    t_te = t_te.reshape(-1, 1)
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)
    ###
    y_unscaled = np.concatenate([y_tr, y_te], axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)
    y_tr = y_scaler.transform(y_tr)
    y_te = y_scaler.transform(y_te)
    train_outputs = []
    test_outputs = []
    targeted_regularization = False
    if dragon == 'donut':
        print('I am here making tarnet')
        net = donut(x_tr.shape[1]).cuda()
        loss = knob_loss
    elif dragon == 'dragonnet':
        print("I am here making dragonnet")
        net = DragonNet(x_tr.shape[1]).cuda()
        targeted_regularization = True
        if targeted_regularization:
            loss = make_tarreg_loss(ratio=1, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss
    elif dragon == 'FE_CFR':
        print("I am here making FE_CFR")
        net = FE_CFR(x_tr.shape[1]).cuda()
        loss = knob_loss
    # for reporducing the IHDP experimemt
    i = 0
    torch.manual_seed(i)
    np.random.seed(i)

    # Get the data and optionally divide into train and test set
    x_train, x_test = x_tr, x_te
    y_train, y_test = y_tr, y_te
    t_train, t_test = t_tr, t_te
    yt_train = np.concatenate([y_train, t_train], 1)
    X_train, X_val, Yt_train, Yt_val = train_test_split(x_train, yt_train, test_size=0.3, random_state=0)
    print(X_train.shape, X_val.shape, Yt_train.shape, Yt_val.shape)
    # Create data loader to pass onto training method
    tensors_train = torch.from_numpy(X_train).float().cuda(), torch.from_numpy(Yt_train).float().cuda()
    tensors_val = torch.from_numpy(X_val).float().cuda(), torch.from_numpy(Yt_val).float().cuda()
    train_loader = DataLoader(TensorDataset(*tensors_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(*tensors_val), batch_size=batch_size)
    import time;
    start_time = time.time()
    # Configuring optimizers
    # Training the networks first for 100 epochs with the Adam optimizer and
    # then for 300 epochs with the SGD optimizer.

    epochs1 = 300
    epochs2 = 300
    early_stopping1 = EarlyStopping(patience=10, verbose=False)
    early_stopping2 = EarlyStopping(patience=20, verbose=False)
    # optimizer_Adam = optim.Adam(net.parameters(), lr=1e-3)
    optimizer_SGD = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    # scheduler_Adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_Adam, mode='min', factor=0.5, patience=2,
    #                                                       threshold=1e-8, cooldown=0, min_lr=0)
    scheduler_SGD = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_SGD, mode='min', factor=0.5, patience=2,
                                                         threshold=0, cooldown=0, min_lr=0)

    train_loss = 0
    valid_loss = 0
    # Adam training run
    # for epoch in range(epochs1):
    #     # Train on data
    #     train_loss = train(train_loader, net, optimizer_Adam, loss, x_train)
    #     scheduler_Adam.step(train_loss)
    #     valid_loss = valid(val_loader, net, loss, x_train)
    #     print(f'Epoch {epoch+1} \t\t Adam \t\tTraining Loss: {train_loss} \t\t Validation  Loss: {valid_loss} ')
    #     early_stopping1(valid_loss, model=net, path=r'..\..\save_model\IHDP')
    #     if early_stopping1.early_stop:
    #         print("Early stopping")
    #         break

    # print(f"**************Adam loss: {train_loss}")
    # SGD training run
    for epoch in range(epochs2):
        # Train on data
        train_loss = train(train_loader, net, optimizer_SGD, loss, x_train)
        scheduler_SGD.step(train_loss)
        valid_loss = valid(val_loader, net, loss, x_train)
        print(f'Epoch {epoch + 1} \t\t SGD \t\tTraining Loss: {train_loss} \t\t Validation  Loss: {valid_loss} ')
        # ==================early stopping======================
        early_stopping2(valid_loss, model=net, path=r'..\..\save_model\IHDP')
        if early_stopping2.early_stop:
            print("Early stopping")
            break
    # print(f"***************SGD loss: {train_loss}")
    elapsed_time = time.time() - start_time
    print("***************************** elapsed_time is: ", elapsed_time)
    net.eval()
    yt_hat_test = net(torch.from_numpy(x_test).float().cuda())
    yt_hat_train = net(torch.from_numpy(x_train).float().cuda())
    test_outputs += [_split_output(yt_hat_test, t_test, y_test, y_scaler, x_test, mu_0_te, mu_1_te, index="TEST", is_targeted_regularization = targeted_regularization)]
    train_outputs += [_split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, mu_0_tr, mu_1_tr, index="TRAIN", is_targeted_regularization = targeted_regularization)]
    return test_outputs, train_outputs


def run_ihdp(data_base_dir='../dat/ihdp/csv', output_dir='../result/ihdp3',
             knob_loss=TTELoss,
             ratio=1., dragon=''):
    print("the model is {}".format(dragon))
    train_cv = np.load(os.path.join(data_base_dir, 'ihdp_npci_1-100.train.npz'))
    test = np.load(os.path.join(data_base_dir, 'ihdp_npci_1-100.test.npz'))
    X_tr = train_cv.f.x.copy()
    T_tr = train_cv.f.t.copy()
    YF_tr = train_cv.f.yf.copy()
    YCF_tr = train_cv.f.ycf.copy()
    mu_0_tr = train_cv.f.mu0.copy()
    mu_1_tr = train_cv.f.mu1.copy()

    X_te = test.f.x.copy()
    T_te = test.f.t.copy()
    YF_te = test.f.yf.copy()
    YCF_te = test.f.ycf.copy()
    mu_0_te = test.f.mu0.copy()
    mu_1_te = test.f.mu1.copy()
    # X = np.concatenate([X_tr,X_te],axis=0)
    T = np.concatenate([T_tr, T_te], axis=0)
    YF = np.concatenate([YF_tr, YF_te], axis=0)
    YCF = np.concatenate([YCF_tr, YCF_te], axis=0)
    mu_0_all = np.concatenate([mu_0_tr, mu_0_te], axis=0)
    mu_1_all = np.concatenate([mu_1_tr, mu_1_te], axis=0)

    for idx in range(X_tr.shape[-1]):
        print("++++", idx, "/", X_tr.shape[-1])
        ##################################
        simulation_output_dir = os.path.join(output_dir, str(idx))
        os.makedirs(simulation_output_dir, exist_ok=True)
        ##################################
        t, y, y_cf, mu_0, mu_1 = T[:,idx], YF[:,idx], YCF[:, idx], mu_0_all[:,idx], mu_1_all[:,idx]
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            t=t, y=y, y_cf=y_cf, mu_0=mu_0, mu_1=mu_1)
        ##################################
        t_tr, y_tr, x_tr, mu0tr, mu1tr = T_tr[:, idx], YF_tr[:, idx], X_tr[:, :, idx], mu_0_tr[:, idx], mu_1_tr[:, idx]
        t_te, y_te, x_te, mu0te, mu1te = T_te[:, idx], YF_te[:, idx], X_te[:, :, idx], mu_0_te[:, idx], mu_1_te[:, idx]

        test_outputs, train_output = train_and_predict_dragons(t_tr, y_tr, x_tr, mu0tr, mu1tr,
                                                                t_te, y_te, x_te, mu0te, mu1te,
                                                               output_dir=simulation_output_dir,
                                                               knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                               val_split=0.2, batch_size=256, idx=idx)

        train_output_dir = os.path.join(simulation_output_dir, "baseline")
        os.makedirs(train_output_dir, exist_ok=True)
        # save the outputs of for each split (1 per npz file)
        for num, output in enumerate(test_outputs):
            np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                **output)
        for num, output in enumerate(train_output):
            np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                **output)
def turn_knob(data_base_dir='../dat/ihdp/csv', knob='FE_CFR',
              output_base_dir='../result/ihdp3'):
    output_dir = os.path.join(output_base_dir, knob + "")
    run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='FE_CFR')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, default="../../dat/IHDP100", help="path to directory LBIDD")
    parser.add_argument('--knob', type=str, default='FE_CFR',
                        help="choose model, i.e., FE_CFR or dragonnet")

    parser.add_argument('--output_base_dir', type=str, default="../../result/IHDP", help="directory to save the output")

    args = parser.parse_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir)


if __name__ == '__main__':
    main()
