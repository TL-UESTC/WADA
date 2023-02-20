import sys
sys.path.append("/home/room/WKK/CADA-w/")
from utils import *
from models.models_config import get_model_config
from models.phm_models import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from phm.data_process import *
from phm.train_eval_phm import *

# Cross_Entropy = nn.BCEWithLogitsLoss(reduction='mean')
# Cross_Entropy2 = nn.BCEWithLogitsLoss(reduction='none')

Cross_Entropy = nn.BCELoss(reduction='mean')
Cross_Entropy2 = nn.BCELoss(reduction='none')

seed = 1

method = 'WAL+DANN'
network = 'CNN_RUL'
src_id = ['OC2']
epsilon = 0.01
if src_id == ['OC1']:
    Gpu = '0'
elif src_id == ['OC2']:
    Gpu = '1'
else:
    Gpu = '2'
tgt_id = ['OC1']
# Gpu = '2'
device = torch.device(f'cuda:{Gpu}')

if network == 'CNN_RUL':
    hyper_param={ 'OC1_OC2': {'epochs':100,'batch_size':512,'lr':1e-3,'lr_D':1e-3,'begin':0.6, 'lambda_w':epsilon, 'k_disc':5}, 
                    'OC1_OC3': {'epochs':100,'batch_size':512,'lr':1e-4,'lr_D':1e-4,'begin':0.5, 'lambda_w':epsilon, 'k_disc':5},
                    'OC2_OC1': {'epochs':200,'batch_size':512,'lr':1e-3,'lr_D':1e-3,'begin':0.3, 'lambda_w':epsilon, 'k_disc':10}, 
                    'OC2_OC3': {'epochs':200,'batch_size':512,'lr':1e-3,'lr_D':1e-3,'begin':0.6, 'lambda_w':epsilon, 'k_disc':10},  
                    'OC3_OC1': {'epochs':200,'batch_size':512,'lr':1e-3,'lr_D':1e-3,'begin':0.25, 'lambda_w':epsilon, 'k_disc':5},
                    'OC3_OC2': {'epochs':200,'batch_size':512,'lr':1e-3,'lr_D':1e-3,'begin':0.25, 'lambda_w':epsilon, 'k_disc':10}} 
else:
    hyper_param={ 'OC1_OC2': {'epochs':50,'batch_size':32,'lr':2e-5,'lr_D':1e-4,'begin':0.2, 'lambda_w':epsilon, 'k_disc':5}, 
                    'OC1_OC3': {'epochs':50,'batch_size':32,'lr':2e-5,'lr_D':1e-4,'begin':0.2, 'lambda_w':epsilon, 'k_disc':5},
                    'OC2_OC1': {'epochs':50,'batch_size':32,'lr':2e-5,'lr_D':1e-4,'begin':0.2, 'lambda_w':epsilon, 'k_disc':10}, 
                    'OC2_OC3': {'epochs':50,'batch_size':32,'lr':2e-5,'lr_D':1e-4,'begin':0.3, 'lambda_w':epsilon, 'k_disc':10},  
                    'OC3_OC1': {'epochs':50,'batch_size':16,'lr':2e-5,'lr_D':1e-4,'begin':0.2, 'lambda_w':epsilon, 'k_disc':5},
                    'OC3_OC2': {'epochs':50,'batch_size':16,'lr':2e-5,'lr_D':2e-5,'begin':0.2, 'lambda_w':epsilon, 'k_disc':10}} 

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def claculate_weight(input1, input2, device):
    input2 = input2.squeeze()
    input2 = input2.detach()
    h = torch.abs(input1 - input2)
    # print(h)
    h = h.detach()
    h = h.cpu().numpy()
    h = np.negative(h)
    w = np.exp(h)
    weight = np.append(w, w)
    # print(weight)
    weight = torch.tensor(weight)
    weight = weight.to(device)
    return weight


def calculate_weighted_loss(preds, labels, weight, Epsilon):
    loss1 = torch.mean(weight * Cross_Entropy2(preds, labels))
    loss2 = Cross_Entropy(preds, labels)
    e = torch.tensor(Epsilon)
    e = e.expand(len(weight)).to(device)
    loss = torch.mean((weight + e) * Cross_Entropy2(preds, labels))
    return loss #loss1 + loss2 * Epsilon


class adversarial_loss(object):
    def __init__(self, preds, labels, weight, Epsilon, epoch, PREHEAT_STEPS, selected_model):
        self.preds = preds
        self.labels = labels
        self.weight = weight
        self.Epsilon = Epsilon
        self.epoch = epoch
        self.PREHEAT_STEPS = PREHEAT_STEPS
        self.selected_model = selected_model
    
    def get_loss(self):
        self.preds = torch.clamp(self.preds, 0.1, 0.9)
        if self.selected_model=='DANN' or self.epoch <= self.PREHEAT_STEPS:
            loss = Cross_Entropy(self.preds, self.labels)
        else:
            loss = calculate_weighted_loss(self.preds, self.labels, self.weight, self.Epsilon)   
        return loss
    

def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1'], ['Bearing1_3']
    elif id == "OC2":
        return ['Bearing2_1'], ['Bearing2_6']
    else:
        return ['Bearing3_1'], ['Bearing3_3']


def get_src_ids(id):
    if id == "OC1":
        return ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'], ['Bearing3_3']
    
    
def get_tgt_ids(id):
    if id == "OC1":
        return ['Bearing1_1','Bearing1_2'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1','Bearing2_2'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2'], ['Bearing3_3']    


def get_src_data(id):
    start_time = time.time()
    x = np.load(f"/media/room/新加卷/WKK/data/source_{id}_x.npy")
    y = np.load(f"/media/room/新加卷/WKK/data/source_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Source data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    if id == "OC1":
        return x, y
    elif id == "OC2":
        return x[:911], y[:911] #np.concatenate((x[:911], x[1708:]), 0), np.concatenate((y[:911], y[1708:]), 0)
    else:
        return x, y


def get_tgt_data(id):
    start_time = time.time()
    x = np.load(f"/media/room/新加卷/WKK/data/source_{id}_x.npy")
    y = np.load(f"/media/room/新加卷/WKK/data/source_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Target data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    if id == "OC1":
        return x[:2803], y[:2803]
    elif id == "OC2":
        return x[:1708], y[:1708]
    else:
        return x[:2152], y[:2152]


def get_test_data(id):
    start_time = time.time()
    x = np.load(f"/media/room/新加卷/WKK/data/test_{id}_x.npy")
    y = np.load(f"/media/room/新加卷/WKK/data/test_{id}_y.npy")
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Test data has been loaded, Time: {epoch_mins}m {epoch_secs}s')
    return x, y


def cross_domain_train(params, device, config, model, src_id, tgt_id, norm_id, data_name, network, gpu):
    
    hyper = params[f'{src_id}_{tgt_id}']
    PREHEAT_STEPS = hyper['epochs']
    Epsilon = 0.2
    
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    
    # save_path = f'./trained_models/DA_phm/resnet/{src_id}_{tgt_id}/'

    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path) 


    train_X_s, train_Y_s = get_src_data(src_id)
    train_X_t, train_Y_t = get_tgt_data(tgt_id)
    test_X_t, test_Y_t = get_test_data(tgt_id)
    
    
    train_set_s = MyDataset_new(train_X_s, train_Y_s)
    train_set_t = MyDataset_new(train_X_t, train_Y_t)
    test_set_t = MyDataset_new(test_X_t, test_Y_t)
    
    
    train_loader_s = DataLoader(dataset=train_set_s,
                              batch_size=hyper['batch_size'],
                              shuffle=True,
                              drop_last=True)

    train_loader_t = DataLoader(dataset=train_set_t,
                              batch_size=hyper['batch_size'],
                              shuffle=True,
                              drop_last=True)
    test_loader_t = DataLoader(dataset=test_set_t,
                              batch_size=64,
                              shuffle=False,
                              drop_last=False)


    load_path = f'/home/room/WKK/SFDA-RUL/trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new.pt'
    checkpoint = torch.load(load_path, map_location=f'cuda:{Gpu}')
    
    source_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
    target_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
    
    source_model.load_state_dict(checkpoint['state_dict'])
    target_model.load_state_dict(checkpoint['state_dict'])
    
    print('=' * 89)
    print(f'The ResNet has {count_parameters(source_model):,} trainable parameters')
    print('=' * 89)
    
    target_encoder = target_model.feature_layers
    discriminator = Discriminator().to(device)

    criterion = RMSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    # discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['lr'], betas=(0.5, 0.9))
    # target_optim = torch.optim.AdamW(target_encoder.parameters(), lr=hyper['lr'], betas=(0.5, 0.9), weight_decay=5e-4)
    
    discriminator_optim = torch.optim.SGD(discriminator.parameters(), lr=0.02)
    target_optim = torch.optim.SGD(target_encoder.parameters(), lr=0.02)
    
    src_only_loss, src_only_score, _, _= evaluate(source_model, test_loader_t, criterion, device, tgt_id)
    
    for epoch in range(1, hyper['epochs'] + 1):
        total_loss = 0
        total_accuracy = 0
        target_losses = 0
        start_time = time.time()
        
        target_encoder.train()
        discriminator.train()

        len_source = len(train_loader_s)
        len_target = len(train_loader_t)
        if len_source > len_target:
            num_iter = len_source
        else:
            num_iter = len_target
            
        for batch_idx in range(num_iter):
            if batch_idx % len_source == 0:
                iter_source = iter(train_loader_s)    
            if batch_idx % len_target == 0:
                iter_target = iter(train_loader_t)
            
            source_x, source_y = iter_source.next()
            target_x, target_y = iter_target.next()
            source_x = source_x.to(device).to(torch.float32)    
            source_y = source_y.to(device).to(torch.float32)   
            target_x = target_x.to(device).to(torch.float32)    
            target_y = target_y.to(device).to(torch.float32)   

            set_requires_grad(target_encoder, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(hyper['k_disc']):
                discriminator_optim.zero_grad()
                _, source_features = target_model(source_x)
                target_pred, target_features = target_model(target_x)
                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                                torch.zeros(target_x.shape[0], device=device)])
                preds = discriminator(discriminator_x).squeeze()
                
                if(epoch > PREHEAT_STEPS):
                    weight = claculate_weight(source_y, target_pred, device)
                    weight = weight.detach()
                    # print(weight)
                    dis_critierion2 = nn.BCEWithLogitsLoss(weight=weight)
                    loss = dis_critierion2(preds, discriminator_y) + dis_critierion(preds, discriminator_y)*Epsilon
                    # loss = weighted_bce_loss(preds, discriminator_y, weight, Epsilon, Lambda_local)
                else: 
                    loss = dis_critierion(preds, discriminator_y)
                    
                # loss = dis_critierion(preds, discriminator_y)
                loss.backward()
                discriminator_optim.step()
                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()
                # Train Feature Extractor
            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(config['k_clf']):
                target_optim.zero_grad()
                target_x = target_x.to(device)              
                _, target_features = target_model(target_x)
                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)
                preds = discriminator(target_features).squeeze()
                target_loss = dis_critierion(preds, discriminator_y)
                # Negaative Contrastive Estimtion Loss
                
                #total loss
                loss = target_loss 
                loss.backward()
                target_optim.step()
                target_losses += target_loss.item()

        mean_loss = total_loss / (num_iter * hyper['k_disc'])
        mean_accuracy = total_accuracy / (num_iter * hyper['k_disc'])
        mean_tgt_loss = target_losses / (num_iter * config['k_clf'] * 2)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch :02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Discriminator_loss:{mean_loss} \t Discriminator_accuracy:{mean_accuracy}')
        print(f'target_loss:{mean_tgt_loss}  \t')
        if epoch % 10 == 0:
            test_loss, test_score, _, true_labels = evaluate(target_model, test_loader_t, criterion, device, tgt_id)
            true_labels = sorted(true_labels, reverse=True)
            # print(true_labels)
            list = [test_loss, test_score]
            data = pd.DataFrame([list])
            # data.to_csv('/home/room/WKK/CADA-w/results/convengence.csv',mode='a',header=False,index=False)
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA Score:{test_score}')
            

def cross_domain_train2(params, device, config, model, src_id, tgt_id, norm_id, data_name, network, gpu):
    Start_Time = time.time()
    best_epoch = 0
    best_rmse = 100
    print (f'Domain Adaptation using: {method}')
    print(f'GPU_id: {Gpu}')
    
    hyper = params[f'{src_id}_{tgt_id}']
    begin = hyper['begin']
    PREHEAT_STEPS = hyper['epochs'] * begin
    Epsilon = hyper['lambda_w']
    print(f'Epsilon: {Epsilon}')
    
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    
    # save_path = f'./trained_models/DA_phm/resnet/{src_id}_{tgt_id}/'

    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path) 


    if network == 'Resnet':
        train_X_s, train_Y_s = get_src_data(src_id)
        train_X_t, train_Y_t = get_tgt_data(tgt_id)
        test_X_t, test_Y_t = get_test_data(tgt_id)

        print(train_X_s.shape, train_X_t.shape)
    else:
        src_trains, src_tests = get_src_ids(src_id)
        tgt_trains, tgt_tests = get_tgt_ids(tgt_id)
        train_X_s, train_Y_s = preprocess(select='train', train_bearings=src_trains, test_bearings=src_tests, norm_id=norm_id, data_name=data_name)
        train_X_t, train_Y_t = preprocess(select='train', train_bearings=tgt_trains, test_bearings=tgt_tests, norm_id=norm_id, data_name=data_name)
        test_X_t, test_Y_t = preprocess(select='test', train_bearings=tgt_trains, test_bearings=tgt_tests, norm_id=norm_id, data_name=data_name)


    train_set_s = MyDataset_new(train_X_s, train_Y_s)
    train_set_t = MyDataset_new(train_X_t, train_Y_t)
    test_set_t = MyDataset_new(test_X_t, test_Y_t)
    
    train_loader_s = DataLoader(dataset=train_set_s,
                              batch_size=hyper['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    train_loader_t = DataLoader(dataset=train_set_t,
                              batch_size=hyper['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)
    test_loader_t = DataLoader(dataset=test_set_t,
                              batch_size=64,
                              shuffle=False,
                              num_workers=0,
                              drop_last=False)
    
    if network == 'Resnet':
        load_path = f'/home/room/WKK/SFDA-RUL/trained_models/pretrained_phm/ResNet/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path, map_location=f'cuda:{Gpu}')
        source_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        target_model = model(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        discriminator = Discriminator().to(device)
        # source_model.load_state_dict(checkpoint['state_dict'])
        # target_model.load_state_dict(checkpoint['state_dict'])
    else:
        load_path = f'/home/room/WKK/CADA-w/trained_models/pretrain_phm/CNN_RUL/pretrained_{src_id}_new.pt'
        checkpoint = torch.load(load_path, map_location=f'cuda:{Gpu}')
        source_model = CNN_RUL().to(device)
        target_model = CNN_RUL().to(device)
        discriminator = Discriminator2().to(device)
        # source_model.load_state_dict(checkpoint['state_dict'])
    
    print('=' * 89)
    print(f'The Model has {count_parameters(target_model):,} trainable parameters')
    print('=' * 89)
    
    target_encoder = target_model.feature_layers
    
    criterion = RMSELoss()
    criterion2 = nn.MSELoss()
    dis_critierion = nn.BCEWithLogitsLoss()
    # optimizer
    lr_t = hyper['lr']
    lr_d = hyper['lr_D']
    discriminator_optim = torch.optim.AdamW(discriminator.parameters(), lr=hyper['lr_D'], betas=(0.5, 0.9))
    target_optim = torch.optim.AdamW(target_model.parameters(), lr=hyper['lr'], betas=(0.5, 0.9), weight_decay=5e-4)
    
    # discriminator_optim = torch.optim.SGD(discriminator.parameters(), lr=0.02)
    # target_optim = torch.optim.SGD(target_encoder.parameters(), lr=0.02)
    
    scheduler_d = StepLR(discriminator_optim, step_size=50, gamma=0.5)
    scheduler_e = StepLR(target_optim, step_size=50, gamma=0.5)
    
    src_only_loss, src_only_mae, src_only_score, _, _= evaluate(source_model, test_loader_t, criterion, device, tgt_id, config)
    
    for epoch in range(1, hyper['epochs'] + 1):
        total_loss = 0
        total_accuracy = 0
        target_losses, rul_losses = 0, 0
        start_time = time.time()
        target_encoder.train()
        discriminator.train()

        len_source = len(train_loader_s)
        len_target = len(train_loader_t)
        if len_source > len_target:
            num_iter = len_source
        else:
            num_iter = len_target
            
        for batch_idx in range(num_iter):
            if batch_idx % len_source == 0:
                iter_source = iter(train_loader_s)    
            if batch_idx % len_target == 0:
                iter_target = iter(train_loader_t)
            
            source_x, source_y = iter_source.next()
            target_x, target_y = iter_target.next()
            source_x = source_x.to(device).to(torch.float32)    
            source_y = source_y.to(device).to(torch.float32)   
            target_x = target_x.to(device).to(torch.float32)    
            target_y = target_y.to(device).to(torch.float32)   

            if config['permute']==True:
                source_x = source_x.permute(0, 2, 1) # permute for CNN model
                target_x = target_x.permute(0, 2, 1)

            set_requires_grad(target_encoder, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=True)
            
            discriminator_optim.zero_grad()
            target_optim.zero_grad()
            
            source_pred, source_features = target_model(source_x)              
            target_pred, target_features = target_model(target_x)
            discriminator_x = torch.cat([source_features, target_features])
            discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                            torch.zeros(target_x.shape[0], device=device)])
            preds = discriminator(discriminator_x).squeeze()
            weight = claculate_weight(source_y, target_pred, device)
            # w = np.random.rand(512)
            # w = np.append(w, w)
            # weight = torch.tensor(w).to(device)
            weight = weight.detach()

            loss = adversarial_loss(preds, discriminator_y, weight, Epsilon, epoch, PREHEAT_STEPS, selected_model=method).get_loss()

            total_loss += loss.item()
            total_accuracy += ((preds > 0.5).long() == discriminator_y.long()).float().mean().item()

            rul_loss = criterion2(source_pred.squeeze(), source_y)
            loss = loss + rul_loss 

            #total loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 2)
            target_optim.step()
            for i in range(4):
                discriminator_optim.step()
            discriminator_optim.step()
            rul_losses += rul_loss.item()
        # scheduler_d.step()
        # scheduler_e.step()
        
        mean_loss = total_loss / num_iter
        mean_accuracy = total_accuracy / num_iter
        mean_rul_loss = rul_losses / num_iter
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch :02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Adv_loss:{mean_loss} \t Discriminator_accuracy:{mean_accuracy}')
        print(f'mean_rul_loss:{mean_rul_loss}')
        if epoch % 5 == 0:
            test_loss, test_mae, test_score, _, true_labels = evaluate(target_model, test_loader_t, criterion, device, tgt_id, config)
            if test_loss < best_rmse:
                best_rmse = test_loss
                best_mae  = test_mae
                best_score = test_score
                best_epoch = epoch
                checkpoint = {'model': target_model,
                               'epoch': epoch,
                               'state_dict': target_model.state_dict()}
            true_labels = sorted(true_labels, reverse=True)
            # print(true_labels)
            list = [test_loss, test_score]
            data = pd.DataFrame([list])
            # data.to_csv('/home/room/WKK/CADA-w/results/convengence.csv',mode='a',header=False,index=False)
            print(f'Src_Only RMSE:{src_only_loss} \t Src_Only MAE:{src_only_mae} \t Src_Only Score:{src_only_score}')
            print(f'DA RMSE:{test_loss} \t DA MAE:{test_mae} \t DA Score:{test_score}')
            
    print(f'Best Epoch:{best_epoch} \t Best RMSE:{best_rmse} \t Best MAE:{best_mae} \t Best Score:{best_score}')
    End_Time = time.time()
    epoch_mins, epoch_secs = epoch_time(Start_Time, End_Time)
    print(f'All Time: {epoch_mins}m {epoch_secs}s')
    
    save_path = save_path = f'./trained_models/DA_phm/resnet/{src_id}_{tgt_id}/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path) 
    torch.save(checkpoint, save_path+'best_model')
    
    save_name = f"/home/room/WKK/CADA-w/results/{network}/results_{method}_{src_id}_{tgt_id}.txt"
    f = open(save_name, 'a')
    f.write(f'Task: {src_id} -> {tgt_id} \t All Time: {epoch_mins}m {epoch_secs}s \t Best Epoch:{best_epoch} \t Best RMSE:{best_rmse} \t Best MAE:{best_mae} \t Bast Score:{best_score} \t Last Epoch:{epoch} \t Last RMSE:{test_loss} \t Last MAE:{test_mae} \t Last Score:{test_score} \t Begin:{begin} \t Epsilon:{Epsilon} \t seed:{seed} \t lr_t:{lr_t} \t lr_d:{lr_d}\n')
    f.close()
            
if __name__ == "__main__":
    layers = 3
    norm_id = 'None'
    data_name = 'phm_data'
    gpu = False
    
    for sid in src_id:
        for tid in tgt_id:
            if sid != tid:
                for _ in range(1):
                    # configuration setup
                    if network == 'CNN_RUL':
                        config = get_model_config('CNN')
                    else:
                        config = get_model_config('LSTM')
                    config.update({'num_runs':1, 'save':False, 'tensorboard':False,'tsne':False,'tensorboard_epoch':False, 'k_disc':5, 'k_clf':1,'iterations':1, 'layers':layers})
                    
                    if method=='ADDA':
                        cross_domain_train(hyper_param, device, config, ResNetFc, sid, tid, norm_id, data_name, network, gpu)
                    else:
                        cross_domain_train2(hyper_param, device, config, ResNetFc, sid, tid, norm_id, data_name, network, gpu)