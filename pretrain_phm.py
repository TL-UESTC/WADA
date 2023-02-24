from concurrent.futures import process
import sys
import torch
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from train_eval_phm import *
from models.models_config import get_model_config
from models.phm_models import *
from data_process import *
from torch.utils.data import DataLoader
device = torch.device('cuda:2')

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def get_ids(id):
    if id == "OC1":
        return ['Bearing1_1','Bearing1_2','Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7'], ['Bearing1_7']
    elif id == "OC2":
        return ['Bearing2_1','Bearing2_2','Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7'], ['Bearing2_6']
    else:
        return ['Bearing3_1', 'Bearing3_2', 'Bearing3_3'], ['Bearing3_3']
    
    
def minmaxscalar(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min)/(max-min)


def pre_train(model, train_dl, test_dl, data_id, config, params, device, save_path):
    # criteierion
      
    criterion = nn.MSELoss()
    criterion2 = RMSELoss()
    criterion3 = nn.L1Loss()
    optimizer_Adam = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=5e-4)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.8)
    optimizer_RMS = torch.optim.RMSprop(model.parameters(), lr=params['lr'], alpha=0.99, eps=1e-4, weight_decay=0, momentum=0, centered=False)
    scheduler = StepLR(optimizer_Adam, step_size=50, gamma=0.5)

    for epoch in range(1, params['pretrain_epoch']+1):
        start_time = time.time()
        train_loss, _, _ = train(model, train_dl, optimizer_Adam, criterion, config, device)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch :02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        # Evaluate on the test set
        if epoch % 5 == 0:
            test_loss, mae, score, _, _ = evaluate(model, test_dl, criterion2, device, data_id, config)
            print('=' * 89)
            print(f'\t  Performance on test set::: Loss: {test_loss:.4f}')
            print(f'\t  Performance on test set::: Score: {score:.4f}')
        
        if params['save']:
            if epoch + 10 == params['pretrain_epoch']:
                checkpoint1 = {'model': model,
                               'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer_Adam.state_dict()}
                torch.save(checkpoint1,
                           save_path + f'pretrained_{data_id}_new2.pt')
            if epoch == params['pretrain_epoch']:
                checkpoint1 = {'model': model,
                               'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optimizer': optimizer_Adam.state_dict()}
                torch.save(checkpoint1,
                           save_path + f'pretrained_{data_id}_new.pt')
    # Evaluate on the test set
    test_loss, mae, score, _, _ = evaluate(model, test_dl, criterion2, device, data_id, config)
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.4f}')
    print(f'\t  Performance on test set:{data_id}::: Score: {score:.4f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return model

if __name__ == '__main__':
    
    data_id = 'OC2'
    norm_id = "None"
    network = "CNN_RUL"
    data_name = 'phm_data'
    gpu = False
    
    if network == 'CNN_RUL':
        config = get_model_config('CNN')
    else:
        config = get_model_config('LSTM')
    
    # config.update({'layers':3})    
    
    train_bearings, test_bearings = get_ids(data_id)
    
    if network == "CNN_RUL":
        train_X, train_Y = preprocess(select='train', train_bearings=train_bearings, test_bearings=test_bearings, norm_id=norm_id, data_name=data_name)
        test_X, test_Y = preprocess(select='test', train_bearings=train_bearings, test_bearings=test_bearings, norm_id=norm_id, data_name=data_name)
    else:
        train_X, train_Y = preprocess_4_ResNet(select='train', train_bearings=train_bearings, test_bearings=test_bearings, data_name=data_name, gpu=gpu)
        test_X, test_Y = preprocess_4_ResNet(select='test', train_bearings=train_bearings, test_bearings=test_bearings, data_name=data_name, gpu=gpu)
    
    normalized = True

    train_set = MyDataset_new(train_X, train_Y)
    test_set = MyDataset_new(test_X, test_Y)
    

    if network=="CNN_RUL":
        hyper_param={ 'OC1': {'pretrain_epoch':200,'batch_size':512,'lr':1e-3, 'save':True, 'tensor_board':False},
                        'OC2': {'pretrain_epoch':200,'batch_size':512,'lr':1e-3, 'save':True, 'tensor_board':False},                  
                        'OC3': {'pretrain_epoch':200,'batch_size':512,'lr':1e-3, 'save':True, 'tensor_board':False}}
    
        hyper = hyper_param[data_id]                
        model = CNN_RUL().to(device)
        save_path = "trained_models/pretrain_phm/CNN_RUL/"
        
    else:
        hyper_param={ 'OC1': {'pretrain_epoch':200,'batch_size':16,'lr':5e-5, 'save':True, 'tensor_board':False},
                        'OC2': {'pretrain_epoch':200,'batch_size':16,'lr':5e-5, 'save':True, 'tensor_board':False},                  
                        'OC3': {'pretrain_epoch':175,'batch_size':16,'lr':5e-5, 'save':True, 'tensor_board':False}}
        

        hyper = hyper_param[data_id]
        model = ResNetFc(resnet_name="ResNet50", use_bottleneck=True, bottleneck_dim=256, new_cls=True, class_num=1).to(device)
        save_path = "trained_models/pretrained_phm/ResNet/"
    
    
    train_loader = DataLoader(dataset=train_set,
                                batch_size=hyper['batch_size'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                            batch_size=100,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False)
    
    pre_train(model, train_loader, test_loader, data_id, config, hyper, device, save_path)