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


def _split_output(yt_hat, x, yt, y_scaler, index, is_targeted_regularization):
    yt_hat = yt_hat.detach().cpu().numpy()
    q_t0 = y_scaler.inverse_transform(yt_hat[:, 0].reshape(-1, 1).copy())
    q_t1 = y_scaler.inverse_transform(yt_hat[:, 1].reshape(-1, 1).copy())
    g = yt_hat[:, 2].copy()

    if is_targeted_regularization:
        eps = yt_hat[:, 3]
    else:
        eps = np.zeros_like(yt_hat[:, 2])
        # eps = np.zeros_like(yt_hat[:, 2])
    y = y_scaler.inverse_transform(yt[:,0].reshape(-1, 1).copy())
    t = yt[:, 1].reshape(-1, 1).copy()
    var = "{}: average propensity for treated: {} and untreated: {}".format(index, g[t.squeeze() == 1.].mean(),
                                                                        g[t.squeeze() == 0.].mean())
    print(var)
    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


def train(train_loader, net, optimizer, criterion, x_train):
    avg_loss = 0
    # iterate through batches
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
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

def train_and_predict_dragons(x, t, y, targeted_regularization=True, output_dir='',
                              knob_loss=TTELoss, ratio=1., dragon='', val_split=0.2, batch_size=256):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # print(f"batch_size: {batch_size}")
    t = t.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ###
    y_scaler = StandardScaler().fit(y)
    y = y_scaler.transform(y)
    train_outputs = []
    test_outputs = []
    targeted_regularization = False
    if dragon == 'donut':
        print('I am here making tarnet')
        net = donut(x.shape[1]).cuda()
        loss = knob_loss
    elif dragon == 'dragonnet':
        print("I am here making dragonnet")
        net = DragonNet(x.shape[1]).cuda()
        targeted_regularization = True
        if targeted_regularization:
            loss = make_tarreg_loss(ratio=1, dragonnet_loss=knob_loss)
        else:
            loss = knob_loss
    elif dragon == 'FE_CFR':
        print("I am here making FE_CFR")
        net = FE_CFR(x.shape[1]).cuda()
        loss = knob_loss
    # for reporducing the IHDP experimemt
    i = 0
    torch.manual_seed(i)
    np.random.seed(i)
    # Get the data and optionally divide into train and test set
    yt_all = np.concatenate([y, t], 1)
    x_train, x_val, yt_train, yt_val = train_test_split(x, yt_all, test_size=0.4, random_state=0, shuffle=True)
    x_val, x_test, yt_val, yt_test = train_test_split(x_val, yt_val, test_size=0.5, random_state=0, shuffle=True)

    print(x_train.shape, x_val.shape, yt_train.shape, yt_val.shape)
    # Create data loader to pass onto training method
    tensors_train = torch.from_numpy(x_train).float().cuda(), torch.from_numpy(yt_train).float().cuda()
    tensors_val = torch.from_numpy(x_val).float().cuda(), torch.from_numpy(yt_val).float().cuda()
    train_loader = DataLoader(TensorDataset(*tensors_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(*tensors_val), batch_size=batch_size)
    import time;
    start_time = time.time()
    # Configuring optimizers
    # Training the networks first for 100 epochs with the Adam optimizer and
    # then for 300 epochs with the SGD optimizer.
    epochs1 = 100
    epochs2 = 300
    early_stopping1 = EarlyStopping(patience=10, verbose=False)
    early_stopping2 = EarlyStopping(patience=20, verbose=False)
    optimizer_Adam = optim.Adam(net.parameters(), lr=1e-3)
    optimizer_SGD = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    scheduler_Adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_Adam, mode='min', factor=0.5, patience=2,
                                                          threshold=1e-8, cooldown=0, min_lr=0)
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
    #     early_stopping1(valid_loss, model=net, path=r'..\..\save_model\Syn_data')
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
    test_outputs += [_split_output(yt_hat_test, x_test, yt_test, y_scaler, index="TEST", is_targeted_regularization = targeted_regularization)]
    train_outputs += [_split_output(yt_hat_train, x_train, yt_train, y_scaler, index="TRAIN", is_targeted_regularization = targeted_regularization)]
    return test_outputs, train_outputs


def run_ihdp(data_base_dir='../dat/ihdp/csv', output_dir='../result/ihdp3', replication=10,
             knob_loss=TTELoss,
             ratio=1., dragon=''):
    print("the model is {}".format(dragon))
    syn_all = np.load(os.path.join(data_base_dir, 'synthetic_data_5.npz'))
    w_all = syn_all['w']
    t_all = syn_all['t']
    y_all = syn_all['y']
    print(w_all.shape, t_all.shape, y_all.shape)
    for idx in range(replication):
        print("++++", idx, "/", replication)
        ##################################
        simulation_output_dir = os.path.join(output_dir, str(idx))
        os.makedirs(simulation_output_dir, exist_ok=True)
        ##################################
        w = w_all[idx,:,:]
        t = t_all[idx, :]
        y = y_all[idx, :]
        print(w.shape, t.shape, y.shape)
        print(t.sum())
        np.savez_compressed(os.path.join(simulation_output_dir, "simulation_outputs.npz"),
                            w=w, t=t, y=y)
        ##################################
        test_outputs, train_output = train_and_predict_dragons(w, t, y,
                                                               output_dir=simulation_output_dir,
                                                               knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                               val_split=0.2, batch_size=1024)
                                                                # batch_size = int(x_tr.shape[0] * 1.0)


        train_output_dir = os.path.join(simulation_output_dir, "baseline")
        os.makedirs(train_output_dir, exist_ok=True)

        # save the outputs of for each split (1 per npz file)
        for num, output in enumerate(test_outputs):
            np.savez_compressed(os.path.join(train_output_dir, "{}_replication_test.npz".format(num)),
                                **output)
        for num, output in enumerate(train_output):
            np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                **output)
def turn_knob(data_base_dir='../dat/ihdp/csv',knob='dragonnet',
              output_base_dir='../result/ihdp3'):
    output_dir = os.path.join(output_base_dir, knob + "")
    run_ihdp(data_base_dir=data_base_dir, output_dir=output_dir, dragon='FE_CFR')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=str, default="../../dat/syn_data", help="path to directory LBIDD")
    parser.add_argument('--knob', type=str, default='FE_CFR',
                        help="choose model, i.e., FE_CFR or dragonnet")
    parser.add_argument('--output_base_dir', type=str, default="../../result/syn-data", help="directory to save the output")
    args = parser.parse_args()
    turn_knob(args.data_base_dir, args.knob, args.output_base_dir)

if __name__ == '__main__':
    main()
