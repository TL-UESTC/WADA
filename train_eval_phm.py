import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn


def phm_score(pred, true):
    # pred = pred[1:]
    # true = true[1:]
    error = (true - pred) / true * 100
    pos_error = error[error > 0]
    neg_error = error[error <= 0]
    score = 0
    for e in neg_error:
        score = math.exp(-math.log(0.5)*(e/5)) + score
    for e in pos_error:
        score = math.exp(math.log(0.5)*(e/20)) + score
    return score / len(true)


def minmaxscalar(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)


def train(model, train_dl, optimizer, criterion, config, device):
    model.train()
    epoch_loss = 0

    for i, data in enumerate(train_dl):
        inputs, labels = data
        src = inputs.to(device).to(torch.float32)
        if config['permute']==True:
            src = src.permute(0, 2, 1).to(device) # permute for CNN model
        labels = labels.to(device).to(torch.float32)
        
        optimizer.zero_grad()
        
        pred, feat = model(src)

        rul_loss = criterion(pred.squeeze(), labels)
        rul_loss.backward()
        
        if (config['model_name']=='LSTM'):
            clip=config['CLIP']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # only for LSTM models
        
        # for name, parms in model.named_parameters():	
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #     ' -->grad_value:',parms.grad)
            
        optimizer.step()
        
        epoch_loss += rul_loss.item()

    return epoch_loss / len(train_dl), pred, labels


def evaluate(model, test_dl, criterion, device, data_id, config):
    mae_loss = nn.L1Loss()
    model.eval()
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_dl):
            inputs, labels = data
            src = inputs.to(device).to(torch.float32)
            if config['permute']==True:
                src = src.permute(0, 2, 1).to(device) # permute for CNN model
            labels = labels.to(device).to(torch.float32)
            pred, feat = model(src)
            
            rul_loss = criterion(pred.squeeze(), labels)
            epoch_loss += rul_loss.item()

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()
            
    test_rmse = criterion(torch.tensor(predicted_rul).to(device), torch.tensor(true_labels).to(device))
    test_mae = mae_loss(torch.tensor(predicted_rul).to(device), torch.tensor(true_labels).to(device))
    score = phm_score(np.array(predicted_rul), np.array(true_labels))
    fig1 = plt.figure()
    plt.plot(predicted_rul, label='pred labels', linewidth=0.5)
    plt.plot(true_labels, label='true labels', linewidth=0.5)
    plt.legend()
    # fig1.savefig('training_images/'+data_id+'.png')
    plt.close(fig1)
    return test_rmse.cpu().numpy(), test_mae.cpu().numpy(), score, predicted_rul, true_labels


def evaluate2(first_model, regressor2, test_dl, criterion, device):
    first_model.eval()
    regressor2.eval()
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_dl):
            inputs, labels = data
            src = inputs.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32)
            pred1, feat = first_model(src)
            pred2 = regressor2(feat)
            pred = (pred1 + pred2) / 2.0
            
            rul_loss = criterion(pred.squeeze(), labels)
            epoch_loss += rul_loss.item()

            predicted_rul += (pred.squeeze().tolist())
            true_labels += labels.tolist()
    test_rmse = criterion(torch.tensor(predicted_rul).to(device), torch.tensor(true_labels).to(device))
    score = phm_score(np.array(predicted_rul), np.array(true_labels))
    fig1 = plt.figure()
    plt.plot(predicted_rul, label='pred labels', linewidth=0.5)
    plt.plot(true_labels, label='true labels', linewidth=0.5)
    plt.legend()
    fig1.savefig('fig1.png')
    plt.close(fig1)
    return test_rmse.cpu().numpy(), score, predicted_rul, true_labels