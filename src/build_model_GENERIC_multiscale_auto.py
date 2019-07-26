import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
import matplotlib.pyplot as plt
from multiscale_convlayer2 import MultiscaleConv2d
import preprocessing_DAIC as pre
import sys, os
import utilities_func as uf
import loadconfig
import configparser

#np.random.seed(0)
#torch.manual_seed(0)
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#load parameters from config file
PREDICTORS_LOAD = cfg.get('model', 'predictors_load')
TARGET_LOAD = cfg.get('model', 'target_load')
TORCH_SAVE_MODEL = cfg.get('model', 'save_model')

#default parameters
dataset = 'daic'
#set correct last-layer dimension
if dataset == 'daic':
    num_classes = 1
channels = 60
gpu_ID = 1
save_best_only = True
early_stopping = False
patience = 10
batch_size = 120
num_epochs = 250
kernel_size_1 = (10,5)
kernel_size_2 = (5, 7)
kernel_size_3 = (3,3)
pool_size = [2,2]
hidden_size = 100
regularization_lambda = 0.001
learning_rate = 0.000001

#look at sys argv: if in crossvalidation model i/o matrices and new model filename
#are given from crossvalidation script, otherwise are normally taken from config.ini
try:
    cross_tag =  sys.argv[1]
    if cross_tag == 'crossvalidation':
        num_experiment = sys.argv[2]
        num_run = sys.argv[3]
        num_fold = sys.argv[4]
        parameters = sys.argv[5]
        model_path = sys.argv[6]
        results_path = sys.argv[7]
        output_temp_data_path = sys.argv[8]
        dataset = sys.argv[9]
        gpu_ID = int(sys.argv[10])
        folds_list = eval(sys.argv[11])
        TORCH_SAVE_MODEL = model_path

        print('crossvalidation mode: I/O from crossvalidation script')
        print('')
        print ('dataset: ' + dataset)
        print ('')
        print('saving model at: ' + TORCH_SAVE_MODEL)
        print ('')

except IndexError:
    print ('regular mode: I/O from config.ini file')
    print ('')
    print ('saving model at: ' + TORCH_SAVE_MODEL)
    print ('')

#path for saving best val loss and best val acc models
BVL_model_path = TORCH_SAVE_MODEL + '_BVL'

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
    for param in parameters:
        exec(param)

except IndexError:
    pass

channels1 = channels
channels2 = channels * 2
channels3 = channels * 3

device = torch.device('cuda:' + str(gpu_ID))
#device = torch.device('cpu')

class EmoModel1layer(nn.Module):

    def __init__(self):
        super(EmoModel1layer, self).__init__()
        self.inner_state = True
        self.conv1 = nn.Conv2d(1, channels, kernel_size=kernel_size_1)
        self.multiscale1 = MultiscaleConv2d(1, channels, kernel_size=kernel_size_1, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.pool = nn.MaxPool2d(pool_size[0], pool_size[1])
        self.hidden = nn.Linear(fc_insize, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        training_state = self.training
        if layer_type == 'conv':
            X = F.relu(self.conv1(X))
        if layer_type == 'multi':
            X = F.relu(self.multiscale1(X, training_state))
        X = X.reshape(X.size(0), -1)
        X = F.relu(self.hidden(X))
        X = self.out(X)

        return X


class EmoModel3layer(nn.Module):

    def __init__(self):
        super(EmoModel3layer, self).__init__()
        self.inner_state = True
        self.conv1 = nn.Conv2d(1, channels1, kernel_size=kernel_size_1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=kernel_size_2)
        self.conv3 = nn.Conv2d(channels2, channels3, kernel_size=kernel_size_3)
        self.multiscale1 = MultiscaleConv2d(1, channels, kernel_size=kernel_size_1, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.multiscale2 = MultiscaleConv2d(channels, channels2, kernel_size=kernel_size_2, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.multiscale3 = MultiscaleConv2d(channels2, channels3, kernel_size=kernel_size_3, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.pool = nn.MaxPool2d(pool_size[0], pool_size[1])
        self.hidden = nn.Linear(fc_insize, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        training_state = self.training
        if layer_type == 'conv':
            X = F.relu(self.conv1(X))
            X = self.pool(X)
            X = F.relu(self.conv2(X))
            X = self.pool(X)
            X = F.relu(self.conv3(X))
            X = self.pool(X)
        if layer_type == 'multi':
            X = F.relu(self.multiscale1(X, training_state))
            X = self.pool(X)
            X = F.relu(self.multiscale2(X, training_state))
            X = self.pool(X)
            X = F.relu(self.multiscale3(X, training_state))
            X = self.pool(X)
        X = X.reshape(X.size(0), -1)
        X = F.relu(self.hidden(X))
        X = self.out(X)

        return X

class EmoModel2layer(nn.Module):

    def __init__(self):
        super(EmoModel2layer, self).__init__()
        self.inner_state = True
        self.conv1 = nn.Conv2d(1, channels1_daic, kernel_size=kernel_size_1_daic)
        self.conv2 = nn.Conv2d(1, channels2_daic, kernel_size=kernel_size_2_daic)
        self.multiscale1 = MultiscaleConv2d(1, channels1_daic, kernel_size=kernel_size_1_daic, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.pool = nn.MaxPool2d(pool_size[0], pool_size[1])
        #self.hidden2 = nn.Linear(fc_insize, 10000)
        #self.hidden3 = nn.Linear(10000, 1000)
        self.hidden4 = nn.Linear(fc_insize, 100)
        self.out = nn.Linear(100, num_classes)

    def forward(self, X):
        training_state = self.training
        if layer_type == 'conv':
            X_time = F.relu(self.conv1(X))
            X_freq = F.relu(self.conv2(X))
        if layer_type == 'multi':
            X_time = F.relu(self.multiscale1(X, training_state))
            X_freq = F.relu(self.multiscale2(X, training_state))
        X_time = self.pool(X_time)
        X_freq = self.pool(X_freq)
        X_time = X_time.reshape(-1)
        X_freq = X_freq.reshape(-1)
        print ('culo')
        print (X_time.shape)
        print (X_freq.shape)
        X = torch.cat((X_time, X_freq))

        X = F.relu(self.hidden4(X))
        X = self.out(X)

        return X


def accuracy(data_x, data_y):
  # calling code must set mode = 'train' or 'eval'
  (max_vals, arg_maxs) = torch.max(data_x.data, dim=1)
  # arg_maxs is tensdataor of indices [0, 1, 0, 2, 1, 1 . . ]
  num_correct = torch.sum(data_y==arg_maxs).float()
  acc = (num_correct * 100.0 / float(len(data_y)))
  return acc.item()  # percentage based

def split_dataset(merged_predictors, merged_target, actors_list, dataset):

    if dataset == 'daic':
        predictors = np.array([])
        target = np.array([])
        for i in actors_list:
            print (i, predictors.shape)
            if i == actors_list[0]:  #if is first item
                predictors = np.array(merged_predictors[i])
                target = np.array(merged_target[i],dtype='float32')
                print (i, predictors.shape)
            else:
                predictors = np.concatenate((predictors, np.array(merged_predictors[i])), axis=0)
                target = np.concatenate((target, np.array(merged_target[i],dtype='float32')), axis=0)

    return predictors, target


def main():

    #CREATE DATASET
    #load numpy data
    print('loading dataset...')

    folds_dataset_path = '../dataset/matrices'
    curr_fold_string = 'daic_test_target_fold_' + str(num_fold) + '.npy'
    curr_fold_path = os.path.join(folds_dataset_path, curr_fold_string)

    train_pred_path = 'daic_training_predictors_fold_' + str(num_fold) + '.npy'
    train_target_path = 'daic_training_target_fold_' + str(num_fold) + '.npy'
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    train_target_path = os.path.join(folds_dataset_path, train_target_path)

    val_pred_path = 'daic_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_target_path = 'daic_validation_target_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    val_target_path = os.path.join(folds_dataset_path, val_target_path)

    test_pred_path = 'daic_test_predictors_fold_' + str(num_fold) + '.npy'
    test_target_path = 'daic_test_target_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)
    test_target_path = os.path.join(folds_dataset_path, test_target_path)

    if not os.path.exists(test_target_path):

        predictors_merged = np.load(PREDICTORS_LOAD)
        target_merged = np.load(TARGET_LOAD)
        predictors_merged = predictors_merged.item()
        target_merged = target_merged.item()

        #split dataset into train, val and test_sets
        train_list = folds_list[int(num_fold)]['train']
        val_list = folds_list[int(num_fold)]['val']
        test_list = folds_list[int(num_fold)]['test']

        training_predictors, training_target = split_dataset(predictors_merged,
                                                            target_merged, train_list, dataset)
        validation_predictors, validation_target = split_dataset(predictors_merged,
                                                            target_merged, val_list, dataset)
        test_predictors, test_target = split_dataset(predictors_merged,
                                                            target_merged, test_list, dataset)


        np.save(train_pred_path, training_predictors)
        np.save(train_target_path, training_target)
        np.save(val_pred_path, validation_predictors)
        np.save(val_target_path, validation_target)
        np.save(test_pred_path, test_predictors)
        np.save(test_target_path, test_target)

    else:
        training_predictors = np.load(train_pred_path)
        training_target = np.load(train_target_path)
        validation_predictors = np.load(val_pred_path)
        validation_target = np.load(val_target_path)
        test_predictors = np.load(test_pred_path)
        test_target = np.load(test_target_path)

    #normalize to 0 mean and unity std (according to training set mean and std)
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)

    #OVERFITTING TEST!!! REMOVE THESE LISES FOR PROPER TRAINING
    validation_predictors = training_predictors.copy()
    validation_target = training_target.copy()

    #normalize labels between 0 and 1
    '''
    max_labels = [np.max(training_target), np.max(validation_target), np.max(test_target)]
    max_val = float(np.max(max_labels))
    training_target = np.divide(training_target, max_val)
    validation_target = np.divide(validation_target, max_val)
    test_target = np.divide(test_target, max_val)
    '''

    #from onehot to float (CrossEntropyLoss requires this)
    '''
    training_target = []
    validation_target = []
    test_target = []
    for i in training_target_onehot:
        training_target.append(np.argmax(i))
    for i in validation_target_onehot:
        validation_target.append(np.argmax(i))
    for i in test_target_onehot:
        test_target.append(np.argmax(i))
    training_target = np.array(training_target)
    validation_target = np.array(validation_target)
    test_target = np.array(test_target)
    '''

    #select a subdataset for testing (to be commented when normally trained)
    '''
    bound = 30
    training_predictors = training_predictors[:bound]
    training_target = training_target[:bound]
    validation_predictors = validation_predictors[:bound]
    validation_target = validation_target[:bound]
    test_predictors = test_predictors[:bound]
    test_target = test_target[:bound]
    '''

    #reshape
    training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
    validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
    test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float()
    val_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    train_target = torch.tensor(training_target).float()
    val_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()

    #build dataset from tensors
    tr_dataset = utils.TensorDataset(train_predictors,train_target) # create your datset
    val_dataset = utils.TensorDataset(val_predictors, val_target) # create your datset
    test_dataset = utils.TensorDataset(test_predictors, test_target) # create your datset

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)  #no batch here!!
    #DNN input shape
    time_dim = training_predictors.shape[1]
    features_dim = training_predictors.shape[2]

    #model = EmoModel((features_dim, time_dim)).to(device)
    if network_type == '1_layer':
        model = EmoModel1layer().to(device)
    if network_type == '2_layer':
        model = EmoModel2layer().to(device)
    if network_type == '3_layer':
        model = EmoModel3layer().to(device)

    #compute number of parameters
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('')
    print ('Total paramters: ' + str(tot_params))

    #define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=regularization_lambda)
    model.train()
    criterion = nn.MSELoss()

    total_step = len(tr_data)
    loss_list = []
    train_loss_hist = []
    val_loss_hist = []
    patience_vec = []

    #TRAINING LOOP
    #iterate epochs
    for epoch in range(num_epochs):
        model.train()
        print ('\n')
        string = 'Epoch: ' + str(epoch+1) + ' '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)
            optimizer.zero_grad()
            outputs = model(sounds)
            loss = criterion(outputs, truth)
            loss.backward()
            #print progress and update history, optimizer step
            perc = int(i / len(tr_data) * 20)
            inv_perc = int(20 - perc - 1)
            loss_print_t = str(np.round(loss.item(), decimals=3))
            string2 = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
            print ('\r', string2, end='')
            optimizer.step()
            #end of batch loop

        #validation loss, training and val accuracy computation
        #after current epoch training
        model.eval()
        train_batch_losses = []
        val_batch_losses = []
        with torch.no_grad():
            #compute training loss
            for i, (sounds, truth) in enumerate(tr_data):
                sounds = sounds.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()
                tr_outputs = model(sounds)
                temp_tr_loss = criterion(tr_outputs, truth)
                train_batch_losses.append(temp_tr_loss.item())

            #compute validation loss
            for i, (sounds, truth) in enumerate(val_data):
                sounds = sounds.to(device)
                truth = truth.to(device)
                optimizer.zero_grad()
                val_outputs = model(sounds)
                temp_val_loss = criterion(val_outputs, truth)
                val_batch_losses.append(temp_val_loss.item())

            #end of epoch loop

        #compute train and val mean loss of current epoch
        train_epoch_loss = np.mean(train_batch_losses)
        val_epoch_loss = np.mean(val_batch_losses)

        #append values to histories
        train_loss_hist.append(train_epoch_loss)
        val_loss_hist.append(val_epoch_loss)

        #print loss and accuracy of the current epoch
        print ('\r', 'train_loss: ' + str(train_epoch_loss) + '| val_loss: ' + str(val_epoch_loss))

        #save best model (metrics = loss)
        if save_best_only == True:
            if epoch == 0:
                torch.save(model.state_dict(), BVL_model_path)
                print ('saved_BVL')
                saved_epoch = epoch + 1
            else:
                best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]
                if curr_loss < best_loss:
                    torch.save(model.state_dict(), BVL_model_path)
                    print ('saved_BVL')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1

        utilstring = 'dataset: ' + str(dataset) + ', exp: ' + str(num_experiment) + ', run: ' + str(num_run) + ', fold: ' + str(num_fold)
        print (utilstring)


        #early stopping
        if early_stopping and epoch >= patience:
            prev_loss = val_hist[-2]
            curr_loss = val_hist[-1]
            if curr_loss < prev_loss:
                patience_vec = []
            else:
                patience_vec.append(curr_loss)
                if len(patience_vec) == patience:
                    print ('\n')
                    print ('Training stopped with patience = ' + str(patience) + ', saved at epoch = ' + str(saved_epoch))
                    break

        #AS LAST THING, AFTER OPTIMIZER.STEP AND EVENTUAL MODEL SAVING
        #AVERAGE MULTISCALE CONV KERNELS!!!!!!!!!!!!!!!!!!!!!!!!!
        if training_mode == 'train_and_eval' or training_mode == 'only_gradient' or training_mode == 'only_train':
            model.multiscale1.update_kernels()
            if network_type == '3_layer':
                model.multiscale2.update_kernels()
                model.multiscale3.update_kernels()
        elif training_mode =='only_eval':
            pass
        else:
            raise NameError ('Invalid training mode')
            print ('Given mode: ' + str(training_mode))

        #END OF EPOCH

    #compute train, val and test loss LOADING the best saved model
    #best validation loss
    #init batch results

    train_batch_losses_BVL = []
    val_batch_losses_BVL = []
    test_batch_losses_BVL = []

    train_batch_preds_BVL = torch.empty(0).to(device)
    val_batch_preds_BVL = torch.empty(0).to(device)
    test_batch_preds_BVL = torch.empty(0).to(device)

    train_batch_truths_BVL = torch.empty(0).to(device)
    val_batch_truths_BVL = torch.empty(0).to(device)
    test_batch_truths_BVL = torch.empty(0).to(device)

    train_stretch_percs_BVL = []
    val_stretch_percs_BVL = []
    test_stretch_percs_BVL = []

    model.load_state_dict(torch.load(BVL_model_path), strict=False)
    model.eval()
    with torch.no_grad():
        #train acc
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
            train_batch_losses_BVL.append(temp_loss)
            train_batch_preds_BVL = torch.cat((train_batch_preds_BVL, temp_pred))
            train_batch_truths_BVL = torch.cat((train_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                train_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())


        #val acc
        for i, (sounds, truth) in enumerate(val_data):
            sounds = sounds.to(device)
            truth = truth.to(device)
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
            val_batch_losses_BVL.append(temp_loss)
            val_batch_preds_BVL = torch.cat((val_batch_preds_BVL, temp_pred))
            val_batch_truths_BVL = torch.cat((val_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                val_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())

        #test acc
        for i, (sounds, truth) in enumerate(test_data):
            sounds = sounds.to(device)
            truth = truth.to(device)
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_loss = criterion(temp_pred, truth)
            test_batch_losses_BVL.append(temp_loss)
            test_batch_preds_BVL = torch.cat((test_batch_preds_BVL, temp_pred))
            test_batch_truths_BVL = torch.cat((test_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                test_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())

    #compute rounded mean of accuracies

    train_loss_BVL = torch.mean(torch.tensor(train_batch_losses_BVL)).cpu().numpy()
    val_loss_BVL = torch.mean(torch.tensor(val_batch_losses_BVL)).cpu().numpy()
    test_loss_BVL = torch.mean(torch.tensor(test_batch_losses_BVL)).cpu().numpy()

    #transfer preds and truths to CPU
    train_batch_preds_BVL = train_batch_preds_BVL.cpu().numpy()
    val_batch_preds_BVL = val_batch_preds_BVL.cpu().numpy()
    test_batch_preds_BVL = torch.tensor(test_batch_preds_BVL).cpu().numpy()

    train_batch_truths_BVL = train_batch_truths_BVL.cpu().numpy()
    val_batch_truths_BVL = val_batch_truths_BVL.cpu().numpy()
    test_batch_truths_BVL = test_batch_truths_BVL.cpu().numpy()

    #process stretch percs
    num_stretches = len(stretch_factors) + 1

    train_stretch_percs_BVL = np.array(train_stretch_percs_BVL)
    train_stretch_percs_BVL = np.mean(train_stretch_percs_BVL, axis=0)
    val_stretch_percs_BVL = np.array(val_stretch_percs_BVL)
    val_stretch_percs_BVL = np.mean(val_stretch_percs_BVL, axis=0)
    test_stretch_percs_BVL = np.array(test_stretch_percs_BVL)
    test_stretch_percs_BVL = np.mean(test_stretch_percs_BVL, axis=0)


    #print results COMPUTED ON THE BEST SAVED MODEL
    print('')
    print ('BVL train LOSS: ' + str(train_loss_BVL))
    print ('BVL val LOSS: ' + str(val_loss_BVL))
    print ('BVL test LOSS: ' + str(test_loss_BVL))

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #save results in temp dict file
    temp_results = {}

    #save loss
    temp_results['train_loss_BVL'] = train_loss_BVL
    temp_results['val_loss_BVL'] = val_loss_BVL
    temp_results['test_loss_BVL'] = test_loss_BVL

    #save preds
    temp_results['train_pred_BVL'] = train_batch_preds_BVL
    temp_results['val_pred_BVL'] = val_batch_preds_BVL
    temp_results['test_pred_BVL'] = test_batch_preds_BVL

    #save truth
    temp_results['train_truth_BVL'] = train_batch_truths_BVL
    temp_results['val_truth_BVL'] = val_batch_truths_BVL
    temp_results['test_truth_BVL'] = test_batch_truths_BVL

    #save history
    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = val_loss_hist
    #save stretch percs
    temp_results['train_stretch_percs_BVL'] = train_stretch_percs_BVL
    temp_results['val_stretch_percs_BVL'] = val_stretch_percs_BVL
    temp_results['test_stretch_percs_BVL'] = test_stretch_percs_BVL

    np.save(results_path, temp_results)



if __name__ == '__main__':
    main()















#
