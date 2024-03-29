## GENERAL DESCRIPTION
This API is aimed at easily defining and running routines
of multiple keras/pytorch trainings and automatically collect the results in a convenient shape. In each instance of a routine (a routine is referred as an 'experiment') it is possible to specify any kind of training-related and model-related parameter, as well as automatically performing k-fold cross-validation. The outcomes of an experiment are saved in a custom-defined folder, which contains:
* A dictionary with all metrics and history, separately computed for every instance of the experiment and for every k-fold.
* All generated models (.hdf5).
* A copy of he current version of the code, saved at the moment of running of the last instance of the experiment.
* A txt file containing a custom description of the experiment.
* A spreadsheet that collects the results of every instance, showing the most important metrics, parameters and highlighting the best results.


## SCRIPTS
* xval_routine: UI to define an experiment and its instances. This script iterates all instances of an experiment calling the script xval_instance.
* xval_instance: This script automatically performs the k-fold cross-validation. It iterates every fold calling the script training.
* training: This script runs the actual trainings.
* define_models: in this script it is possible to define custom keras models.
* utility_functions: some utilities.
* results_to_excel: computes the spreadsheets.
* preprocessing_DATASET: processes audio data building the features matrices calling the script feat_analysis.
* preprocessing_utils: contains the feature extraction and other general utilities to preprocess audio datasets.
* augmentation: applies augmentation to audio data
* gen_data_description: computes a dictionary containing audio descriptors of every sample of a dataset.
* config.ini: This config file contains mainly I/O folder paths and defaults parameters for the training.


## EXPERIMENT DEFINITION
For each experiment you can create a new xval_routine script, copying the example one.
In each experiment it is mandatory to define the following macro parameters:
* A short description of the experiment that will be saved in a txt file. For example 'In this experiment we tested different learning rates'
* gpu_ID: GPU number ID in which run the trainings
* dataset: a short name of the used dataset. This will affect the name of the result files and serves as well to load the correct dataset.
* num_experiment: number of current experiment (has to be an integer).
* num_folds: int, how many k for the k-fold cross-validation.
* experiment_folder: path in which save all result files. Different experiments for the same dataset are saved in the same macro-directory.
* overwrite_results: (bool) False if you want to avoid to overwrite previous results
* debug_mode: (bool) if False, when en error occurs the script will pass to the next instance without stopping.
* task_type: should be 'classification' or 'regression'
* generator: (bool) if True trains with generator



For each experiment, you should define a dictionary containing the instances of the experiment (the keys should be progressive integers). Each key/instance has to be a list of strings and each element of the list should be a parameter declaration statement.

Example:
```python
experiment[1] = ['architecture="EXAMPLE_model"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.001"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.001"']
experiment[2] = ['architecture="EXAMPLE_model"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.01"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.01"']
experiment[3] = ['architecture="EXAMPLE_model"', 'reshaping_type="cnn"',
                 'comment_1="reg 0.1"', 'comment_2="EXAMPLE_architecture"',
                 'regularization_lambda="0.1"']

```

The parameters you insert will overwrite the default ones, which are declared in the config.ini file (parameters related to the training) and the define_models script (parameters related to the very models). In each instance (each key of the dict) it is mandatory to declare at least these parameters (See previous example):
* comment_1 and comment_2: comments that are plotted in the spreadsheet.
* architecture: the model you want to use. Should be the name of a model function present in models_API script.
* reshaping_type: matrix reshaping before being fed into the models. It can be: 'conv', 'rnn' or 'none'. It is coherent with TensorFlow backend.

It is NOT possible to insert parameters relative to the preprocessing stage.

For a quicker usage, it is possible to run an xval_routine script from command line with 3 positional parameters:
1. first instance to run
2. last instance to run
3. gpu ID

Example:
```bash
python3 xval_instance_ravdess_exp1 3 7 1
```
The above code runs instances between 3 and 7 of experiment 1 in the GPU number 1

## CUSTOM MODELS DEFINITION
To define a model simply write a function containing the model and put in in the define_models script. Then you can call your custom model as a simple parameter in the xval_routine script. You must use a predefined structure as in the following example:

```python
#ALL MODEL FUNCTIONS SHOULD HAVE THESE 3 INPUT PARAMETERS:
#time_dim
#features_dim
#user_parameters
def EXAMPLE_model(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the xval_routine script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [16, 12],
    'conv1_depth': 20,
    'hidden_size': 200}

    reg = regularizers.l2(p['regularization_lambda'])

    #THEN CALL parse_parameters() TO OVERWRITE DEFAULT PARAMETERS
    #WITH THE PARAMETERS DEFINED IN THE xval_routine SCRIPT
    p = parse_parameters(p, user_parameters)

    #DECLARE YOUR ARCHITECTURE
    #input dims should be time_dim and features_dim
    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict
    conv_1 = Convolution2D(p['conv1_depth'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    flat = Flatten()(conv_1)
    drop_1 = Dropout(p['drop_prob'])(flat)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(drop_1)
    out = Dense(8, activation='softmax')(hidden)

    #INSTANTIATE THE MODEL
    model = Model(inputs=input_data, outputs=out)

    #FINALLY, ALWAYS RETURN BOTH THE MODEL AND THE PARAMETERS DICT
    return model, p
```


## PREPROCESSING
The preprocessing needs to be customized for every new dataset. In order to be compatible con the rest of the API, any proprocessing script has to output 2 dictionaries: 1 containing the predictors and 1 containing the target. The keys of these dicts has to be the 'foldable items', that is the criterion that you want to use to divide train-validation-test sets, for example the different actors in a speech dataset. So every key should contain all data from one single actor.

Example:
```python
predictors_dict['1': matrix with all spectra of actor 1,
                '2': matrix with all spectra of actor 2]
target_dict['1': matrix with all labels of actor 1,
            '2': matrix with all labels of actor 1]
```

In the end, you should save the predictors and target dicts as:
```
nameOfDataset_featuresType_predictors.npy
nameOfDataset_featuresType_target.npy
```
Example:
```
ravdess_mfcc_predictors.npy
ravdess_mfcc_target.npy
```

Then, you can define nameOfDataset_featuresType as the dataset name in the xval_routine script.

Example:
```python
dataset = ravdess_mfcc
```
This means that you should treat different preprocessings of the same dataset as they were different datasets in the xval_routine interface.

The preprocessing_utils script contains useful functions to help the development of the custom preprocessing for a dataset. See the preprocessing_RAVDESS and preprocessing_IEMOCAP scripts for detailed examples.

In particular, in your preprocessing script you can call the function prepreocessing_utils.preprocess_foldable_item(), which takes these 3 parameters:
1. List of soundpath of current foldable item (example: all paths to sounds of one actor)
2. integer describing the maximum file length (in samples) of the current dataset. This serves to zeropad sounds to the same dimension (only if segmentation parameter is False).
3. Python function that takes as input the path of a soundfile and outputs its label (this should be custom for every dataset)

So, for the custom preprocessing of a dataset you could simply create a for loop that iterates the foldable items and calls the above function.

Example in pseudo-code:
```python
#this is the basic structure to preprocess a new dataset
foldable_items = {1: ["paths to actor 1's sounds"],
                  2: ["paths to actor 2's sounds"],
                  3: ["paths to actor 3's sounds"],
                  ...
                  ...
                  }

def get_label(wavname):
  #func to extract label from file path
  return label

max_file_length = #function that computes this value (you can use utility_functions.find_longest_audio_list(input_list))

#iterate foldable items
actors = list(foldable_items.keys())
predictors = {}
target = {}

for i in actors:
  curr_list = foldable_items[i]
  curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label)
  predictors[i] = curr_predictors
  target[i] = curr_target

save_predictors()
save_target()
```

In the [feature_extraction] of the config.ini file you can set all important parameters for feature extraction.


## AUGMENTATION
The augmentation script applies random transformations in random order to audio files. Audio files should be mono '.wav' sounds of any sample rate. This script loads 1 audio file containing a background noise sample (wav) and 1 folder containing room impulse responses (also wav). The path to these information should be defined in the config.ini file in the [augmentation] section

The transformations are:
* Adding background noise
* Random eq
* Random reverb
* Random time stretch

To augment a dataset you can:
1. Run a routine that creates new audiofiles (and copies the original ones) in a new folder. This is convenient if you want to listen to the augmented samples
2. Call a function that does this 'in place' during the preprocessing. This is convenient if you want to save space and is also simpler.

### Option 1:
Run the augmentation script with the following arguments:
1. input folder
2. output folder
3. number of generated augmented samples for every input file

Example:
```python
run augmentation.py 'dataset/original' '/dataset/augmented' 5  
```
This code takes all mono wav files present in '/dataset/original/', generates 5 augmented samples for each file and saves them, alongside with a normalized copy of the original, in '/dataset/augmented'

### Option 2:
When you call the prepreocessing_utils.preprocess_foldable_item() in your preprocessing script, you can set these parameters in the config.ini file:
```
[feature_extraction]
augmentation = True
num_aug_samples = 2  #this int describes how many augmented samples are created for every original sound of the dataset. 2 means that the size of the dataset it triplicated.
```


## GEN_DATA_DESCRIPTION
The data description algorithm computes the following audio descriptors for every wav file contained in a folder:
* High frequency content
* Spectral centroid
* Pitch
* Pitch salience percentage (how much is present silence in a file)
* Duration
* Duration cutting initial and final silence
* Yell factor (describes hou much a person is speaking loud)

To use this script run it with the following 2 positional arguments:
1. Input folder
2. Output name of the generated dictionary

Example:
```python
run data_description.py '..dataset/ravdess/audio_data' '../dataset/dataset_descriptions/ravdess_description.npy'  
```

The output dictionary will have one key for each audio file, containing all descriptors.

Example:
```python
{'03-01-01-01-01-02-08.wav': {'mean_hfc': 0.0663644,
  'std_hfc': 0.083298385,
  'mean_centroid': 1102.4412,
  'std_centroid': 1137.6855,
  'mean_energy': 0.00010976361,
  'std_energy': 0.00012627964,
  'mean_F0': 358.17438,
  'std_F0': 47.01472,
  'mean_yell_factor': 0.041032843,
  'std_yell_factor': 0.063268915,
  'perc_salience': 0.9390243902439024,
  'dur': 88794,
  'dur_stripped': 42496}}  
```
