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
print("loading dataset...")
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#load parameters from config file
TRAINING_PREDICTORS = cfg.get('model', 'predictors_load')
TRAINING_TARGET = cfg.get('model', 'target_load')
TORCH_SAVE_MODEL = cfg.get('model', 'save_model')

#defaults
dataset = 'daic'


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
        fold_sequence = sys.argv[11]
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
BVA_model_path = TORCH_SAVE_MODEL + '_BVA'

#set correct output classes
if dataset == 'daic':
    num_classes = 1

#global parameters
#gpu_ID = 1
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
learning_rate = 0.001

'''
str to useetch_factors = [(0.8, 1.),(1.25,1.)]  #multiscale stretch
output_type = 'pooled_map'
stretch_penality_lambda = 0.
layer_type = 'conv'
training_mode = 'only_eval'
network_type = "1_layer"
channels1 = 20
channels2 = 28
channels3 = 40
'''

#OVERWRITE PARAMETERS IF IN XVAL MODE
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
        self.conv1 = nn.Conv2d(1, channels1, kernel_size=kernel_size_1)
        self.conv2 = nn.Conv2d(channels1, channels1, kernel_size=kernel_size_1)
        self.multiscale1 = MultiscaleConv2d(1, channels, kernel_size=kernel_size_1, scale_factors=stretch_factors,
                                           output_type=output_type, stretch_penality_lambda= stretch_penality_lambda)
        self.multiscale2 = MultiscaleConv2d(channels1, channels1, kernel_size=kernel_size_1, scale_factors=stretch_factors,
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
        if layer_type == 'multi':
            X = F.relu(self.multiscale1(X, training_state))
            X = self.pool(X)
            X = F.relu(self.multiscale2(X, training_state))
        X = X.reshape(X.size(0), -1)
        X = F.relu(self.hidden(X))
        X = self.out(X)

        return X


def accuracy(data_x, data_y):
  # calling code must set mode = 'train' or 'eval'
  (max_vals, arg_maxs) = torch.max(data_x.data, dim=1)
  # arg_maxs is tensdataor of indices [0, 1, 0, 2, 1, 1 . . ]
  num_correct = torch.sum(data_y==arg_maxs).float()
  acc = (num_correct * 100.0 / float(len(data_y)))
  return acc.item()  # percentage based

def split_dataset(dataset_dict, xval_):
    '''
        elif dataset == 'mnist':
            predictors = np.array([])
            target = np.array([])
            for i in actors_list:
                print (i, predictors.shape)
                if i == actors_list[0]:
                    predictors = merged_predictors[i]
                    target = merged_target[i]
                    print (i, predictors.shape)

                else:
                    predictors = np.concatenate((predictors, merged_predictors[i]), axis=0)
                    target = np.concatenate((target, merged_target[i]), axis=0)


    '''
def main():

    #CREATE DATASET
    print ('culo')
    sys.exit(0)
    #load numpy data for other datasets
    training_predictors = np.load(TRAINING_PREDICTORS)
    training_target_onehot = np.load(TRAINING_TARGET)


    #normalize to 0 mean and unity std (according to training set mean and std)
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)

    #from onehot to float (CrossEntropyLoss requires this)
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
    train_predictors = torch.tensor(training_predictors).float().to(device)
    val_predictors = torch.tensor(validation_predictors).float().to(device)
    test_predictors = torch.tensor(test_predictors).float().to(device)
    train_target = torch.tensor(training_target, dtype=torch.long).to(device)
    val_target = torch.tensor(validation_target, dtype=torch.long).to(device)
    test_target = torch.tensor(test_target, dtype=torch.long).to(device)

    #build dataset from tensors
    tr_dataset = utils.TensorDataset(train_predictors,train_target) # create your datset
    val_dataset = utils.TensorDataset(val_predictors, val_target) # create your datset
    test_dataset = utils.TensorDataset(test_predictors, test_target) # create your datset

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False)  #no batch here!!
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
    criterion = nn.CrossEntropyLoss()

    total_step = len(tr_data)
    loss_list = []
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    patience_vec = []

    #TRAINING LOOP
    #iterate epochs
    for epoch in range(num_epochs):
        model.train()
        print ('\n')
        string = 'Epoch: ' + str(epoch+1) + ' '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
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
        train_batch_accs = []
        val_batch_losses = []
        val_batch_accs = []
        with torch.no_grad():
            #compute training accuracy and loss
            for i, (sounds, truth) in enumerate(tr_data):
                optimizer.zero_grad()
                tr_outputs = model(sounds)
                temp_tr_loss = criterion(tr_outputs, truth)
                train_batch_losses.append(temp_tr_loss.item())
                temp_tr_acc = accuracy(tr_outputs, truth)
                train_batch_accs.append(temp_tr_acc)
            #compute validation accuracy and loss
            for i, (sounds, truth) in enumerate(val_data):
                optimizer.zero_grad()
                val_outputs = model(sounds)
                temp_val_loss = criterion(val_outputs, truth)
                val_batch_losses.append(temp_val_loss.item())
                temp_val_acc = accuracy(val_outputs, truth)
                val_batch_accs.append(temp_val_acc)
            #end of epoch loop

        #compute train and val mean accuracy and loss of current epoch
        train_epoch_loss = np.mean(train_batch_losses)
        train_epoch_acc = np.mean(train_batch_accs)
        val_epoch_loss = np.mean(val_batch_losses)
        val_epoch_acc = np.mean(val_batch_accs)

        #append values to histories
        train_loss_hist.append(train_epoch_loss)
        train_acc_hist.append(train_epoch_acc)
        val_loss_hist.append(val_epoch_loss)
        val_acc_hist.append(val_epoch_acc)


        #print loss and accuracy of the current epoch
        print ('\n', 'train_acc: ' + str(train_epoch_acc) + ' | val_acc: ' + str(val_epoch_acc))
        print ('\r', 'train_loss: ' + str(train_epoch_loss) + '| val_loss: ' + str(val_epoch_loss))

        #save best model (metrics = loss)
        if save_best_only == True:
            if epoch == 0:
                torch.save(model.state_dict(), BVL_model_path)
                torch.save(model.state_dict(), BVA_model_path)
                print ('saved_BVL')
                print ('saved_BVA')
                saved_epoch = epoch + 1
            else:
                best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
                best_acc = max(val_acc_hist[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]
                curr_acc = val_acc_hist[-1]
                if curr_loss < best_loss:
                    torch.save(model.state_dict(), BVL_model_path)
                    print ('saved_BVL')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1

                if curr_acc > best_acc:
                    torch.save(model.state_dict(), BVA_model_path)
                    print ('saved_BVA')  #SUBSTITUTE WITH SAVE MODEL FUNC
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

    #compute train, val and test accuracy LOADING the best saved model
    #best validation loss
    #init batch results
    train_batch_accs_BVL = []
    val_batch_accs_BVL =[]
    test_batch_accs_BVL = []

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
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            train_batch_accs_BVL.append(temp_acc)
            train_batch_losses_BVL.append(temp_loss)
            train_batch_preds_BVL = torch.cat((train_batch_preds_BVL, temp_pred))
            train_batch_truths_BVL = torch.cat((train_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                train_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())
        #val acc
        for i, (sounds, truth) in enumerate(val_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            val_batch_accs_BVL.append(temp_acc)
            val_batch_losses_BVL.append(temp_loss)
            val_batch_preds_BVL = torch.cat((val_batch_preds_BVL, temp_pred))
            val_batch_truths_BVL = torch.cat((val_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                val_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())
        #test acc
        for i, (sounds, truth) in enumerate(test_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            test_batch_accs_BVL.append(temp_acc)
            test_batch_losses_BVL.append(temp_loss)
            test_batch_preds_BVL = torch.cat((test_batch_preds_BVL, temp_pred))
            test_batch_truths_BVL = torch.cat((test_batch_truths_BVL, truth.float()))
            if layer_type == 'multi':
                test_stretch_percs_BVL.append(model.multiscale1.get_stretch_percs())
    #best_validation_accuracy
    #init batch results
    train_batch_accs_BVA = []
    val_batch_accs_BVA =[]
    test_batch_accs_BVA = []

    train_batch_losses_BVA = []
    val_batch_losses_BVA = []
    test_batch_losses_BVA = []

    train_batch_preds_BVA = torch.empty(0).float().to(device)
    val_batch_preds_BVA = torch.empty(0).float().to(device)
    test_batch_preds_BVA = torch.empty(0).float().to(device)

    train_batch_truths_BVA = torch.empty(0).float().to(device)
    val_batch_truths_BVA = torch.empty(0).float().to(device)
    test_batch_truths_BVA = torch.empty(0).float().to(device)

    train_stretch_percs_BVA = []
    val_stretch_percs_BVA = []
    test_stretch_percs_BVA = []

    model.load_state_dict(torch.load(BVA_model_path), strict=False)
    model.eval()
    with torch.no_grad():
        #train acc
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            train_batch_accs_BVA.append(temp_acc)
            train_batch_losses_BVA.append(temp_loss)

            train_batch_preds_BVA = torch.cat((train_batch_preds_BVA, temp_pred))
            train_batch_truths_BVA = torch.cat((train_batch_truths_BVA, truth.float()))
            if layer_type == 'multi':
                train_stretch_percs_BVA.append(model.multiscale1.get_stretch_percs())
        #val acc
        for i, (sounds, truth) in enumerate(val_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            val_batch_accs_BVA.append(temp_acc)
            val_batch_losses_BVA.append(temp_loss)
            val_batch_preds_BVA = torch.cat((val_batch_preds_BVA, temp_pred))
            val_batch_truths_BVA = torch.cat((val_batch_truths_BVA, truth.float()))
            if layer_type == 'multi':
                val_stretch_percs_BVA.append(model.multiscale1.get_stretch_percs())
        #test acc
        for i, (sounds, truth) in enumerate(test_data):
            optimizer.zero_grad()
            temp_pred = model(sounds)
            temp_acc = accuracy(temp_pred, truth)
            temp_loss = criterion(temp_pred, truth)
            test_batch_accs_BVA.append(temp_acc)
            test_batch_losses_BVA.append(temp_loss)
            test_batch_preds_BVA = torch.cat((test_batch_preds_BVA, temp_pred))
            test_batch_truths_BVA = torch.cat((test_batch_truths_BVA, truth.float()))
            if layer_type == 'multi':
                test_stretch_percs_BVA.append(model.multiscale1.get_stretch_percs())
    #compute rounded mean of accuracies
    train_acc_BVL = np.mean(train_batch_accs_BVL)
    val_acc_BVL = np.mean(val_batch_accs_BVL)
    test_acc_BVL = np.mean(test_batch_accs_BVL)

    train_acc_BVA = np.mean(train_batch_accs_BVA)
    val_acc_BVA = np.mean(val_batch_accs_BVA)
    test_acc_BVA = np.mean(test_batch_accs_BVA)

    train_loss_BVL = torch.mean(torch.tensor(train_batch_losses_BVL)).cpu().numpy()
    val_loss_BVL = torch.mean(torch.tensor(val_batch_losses_BVL)).cpu().numpy()
    test_loss_BVL = torch.mean(torch.tensor(test_batch_losses_BVL)).cpu().numpy()

    train_loss_BVA = torch.mean(torch.tensor(train_batch_losses_BVA)).cpu().numpy()
    val_loss_BVA = torch.mean(torch.tensor(val_batch_losses_BVA)).cpu().numpy()
    test_loss_BVA = torch.mean(torch.tensor(test_batch_losses_BVA)).cpu().numpy()

    #transfer preds and truths to CPU
    train_batch_preds_BVL = train_batch_preds_BVL.cpu().numpy()
    val_batch_preds_BVL = val_batch_preds_BVL.cpu().numpy()
    test_batch_preds_BVL = torch.tensor(test_batch_preds_BVL).cpu().numpy()

    train_batch_truths_BVL = train_batch_truths_BVL.cpu().numpy()
    val_batch_truths_BVL = val_batch_truths_BVL.cpu().numpy()
    test_batch_truths_BVL = test_batch_truths_BVL.cpu().numpy()

    train_batch_preds_BVA = train_batch_preds_BVA.cpu().numpy()
    val_batch_preds_BVA = val_batch_preds_BVA.cpu().numpy()
    test_batch_preds_BVA = test_batch_preds_BVA.cpu().numpy()

    train_batch_truths_BVA = train_batch_truths_BVA.cpu().numpy()
    val_batch_truths_BVA = val_batch_truths_BVA.cpu().numpy()
    test_batch_truths_BVA = test_batch_truths_BVA.cpu().numpy()

    #process stretch percs
    num_stretches = len(stretch_factors) + 1

    train_stretch_percs_BVL = np.array(train_stretch_percs_BVL)
    train_stretch_percs_BVL = np.mean(train_stretch_percs_BVL, axis=0)
    val_stretch_percs_BVL = np.array(val_stretch_percs_BVL)
    val_stretch_percs_BVL = np.mean(val_stretch_percs_BVL, axis=0)
    test_stretch_percs_BVL = np.array(test_stretch_percs_BVL)
    test_stretch_percs_BVL = np.mean(test_stretch_percs_BVL, axis=0)

    train_stretch_percs_BVA = np.array(train_stretch_percs_BVA)
    train_stretch_percs_BVA = np.mean(train_stretch_percs_BVA, axis=0)
    val_stretch_percs_BVA = np.array(val_stretch_percs_BVA)
    val_stretch_percs_BVA = np.mean(val_stretch_percs_BVA, axis=0)
    test_stretch_percs_BVA = np.array(test_stretch_percs_BVA)
    test_stretch_percs_BVA = np.mean(test_stretch_percs_BVA, axis=0)


    #print results COMPUTED ON THE BEST SAVED MODEL
    print('')
    print ('BVL train acc: ' + str(train_acc_BVL))
    print ('BVL val acc: ' + str(val_acc_BVL))
    print ('BVL test acc: ' + str(test_acc_BVL))

    print ('BVA train acc: ' + str(train_acc_BVA))
    print ('BVA val acc: ' + str(val_acc_BVA))
    print ('BVA test acc: ' + str(test_acc_BVA))

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    #save results in temp dict file
    temp_results = {}
    #save accuracy
    temp_results['train_acc_BVL'] = train_acc_BVL
    temp_results['val_acc_BVL'] = val_acc_BVL
    temp_results['test_acc_BVL'] = test_acc_BVL
    temp_results['train_acc_BVA'] = train_acc_BVA
    temp_results['val_acc_BVA'] = val_acc_BVA
    temp_results['test_acc_BVA'] = test_acc_BVA
    #save loss
    temp_results['train_loss_BVL'] = train_loss_BVL
    temp_results['val_loss_BVL'] = val_loss_BVL
    temp_results['test_loss_BVL'] = test_loss_BVL
    temp_results['train_loss_BVA'] = train_loss_BVA
    temp_results['val_loss_BVA'] = val_loss_BVA
    temp_results['test_loss_BVA'] = test_loss_BVA
    #save preds
    temp_results['train_pred_BVL'] = train_batch_preds_BVL
    temp_results['val_pred_BVL'] = val_batch_preds_BVL
    temp_results['test_pred_BVL'] = test_batch_preds_BVL
    temp_results['train_pred_BVA'] = train_batch_preds_BVA
    temp_results['val_pred_BVA'] = val_batch_preds_BVA
    temp_results['test_pred_BVA'] = test_batch_preds_BVA
    #save truth
    temp_results['train_truth_BVL'] = train_batch_truths_BVL
    temp_results['val_truth_BVL'] = val_batch_truths_BVL
    temp_results['test_truth_BVL'] = test_batch_truths_BVL
    temp_results['train_truth_BVA'] = train_batch_truths_BVA
    temp_results['val_truth_BVA'] = val_batch_truths_BVA
    temp_results['test_truth_BVA'] = test_batch_truths_BVA
    #save history
    temp_results['train_loss_hist'] = train_loss_hist
    temp_results['val_loss_hist'] = val_loss_hist
    temp_results['train_acc_hist'] = train_acc_hist
    temp_results['val_acc_hist'] = val_acc_hist
    #save stretch percs
    temp_results['train_stretch_percs_BVL'] = train_stretch_percs_BVL
    temp_results['val_stretch_percs_BVL'] = val_stretch_percs_BVL
    temp_results['test_stretch_percs_BVL'] = test_stretch_percs_BVL
    temp_results['train_stretch_percs_BVA'] = train_stretch_percs_BVA
    temp_results['val_stretch_percs_BVA'] = val_stretch_percs_BVA
    temp_results['test_stretch_percs_BVA'] = test_stretch_percs_BVA


    np.save(results_path, temp_results)



if __name__ == '__main__':
    main()















#
