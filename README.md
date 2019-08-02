## GENERAL DESCRIPTION
This API is aimed at easily defining and running routines
of multiple keras/pytorch trainings and automatically collect the results in a convenient shape. In each instance of a routine (called 'experiment') it is possible to specify any kind of training-related and model-related parameter, as well as automatically performing k-fold cross-validation. The outcomes of an experiment are saved in a custom-defined folder, which contains:
* A dictionary with all metrics and history, separately computed for every instance of the experiment and for every k-fold.
* All generated models (.hdf5).
* A copy of he current version of the code, saved at the moment of running of the last instance of the experiment.
* A txt file containing a custom description of the experiment.
* A spreadsheet that collects the results of every instance, showing the most important metrics, parameters and highlighting the best results.


## SCRIPTS
* xval_routine: UI to define an experiment and its instances. This script iterates all instances of an experiment calling the script xval_instance.
* xval_instance: This script automatically performs the k-fold cross-validation. It iterates every fold calling the script build_model.
* training: This script runs the actual trainings.
* define_models: in this script it is possible to define custom keras models.
* utility_functions: self-explanatory.
* results_to_excel: computes the spreadsheets.
* preprocessing_DATASET: processes audio data building the features matrices calling the script feat_analysis.
* feat_analysis: contains the feature extraction functions: STFT and MFCC
* config.ini: This config file contains mainly I/O folder paths and defaults parameters for the training


## EXPERIMENT DEFINITION
For each experiment you can create a new xval_routine script, copying the example one.
In each experiment it is mandatory to define these macro parameters
* A short description of the experiment that will be saved in a txt file. For example 'In this experiment we tested different learning rates'
* gpu_ID: GPU number ID in which run the training
* dataset: a short name of the used dataset. This will affect the name of the results and serves as well to load the correct dataset.
* num_experiment: number of current experiment (has to be an integer).
* num_folds: int, how many k for the k-fold cross-validation.
* experiment_folder: path in which save all results. Different experiments for the same dataset are saved in the same directory.
* overwrite_results: False if you want to avoid overwriting previous results
* debug_mode: if False, when en error occurs the script will pass to the next instance without stopping.
In each experiment you should define a dict containing the instances of the experiment. The keys should be progressive integers.
Each key/instance has to be a list of strings and each element of a list is a parameter declaration.

Example:
```python
experiment_dict[1] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg base 0.001"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.001"']
experiment_dict[2] = ['task_type= "classification"', 'architecture="EXAMPLE_model"',
                 'comment_1="reg increased 0.01"', 'comment_2="EXAMPLE_architecture"','regularization_lambda="0.01"']

```

The parameters you insert overwrite the default one, which are declared in the build_model and models_API scripts. Since a copy of the code issaved for every experiment, you can easily check which were the default parameters in case you change them.
In each instance it is mandatory to declare at least these 4 parameters (See previous example):
* comment_1 and comment_2: comments that are plotted in the spreadsheet.
* task_type: should be 'classification' or 'regression'
* architecture: the model you want to use. Should be the name of a model function present in models_API script.

It is possible to run an xval_routine script from the terminal with 3 positional parameters:
1: first instance to run
2: last instance to run
3: gpu ID

Example:
```bash
python3 xval_instance_ravdess_exp1 3 7 1
```
The above code runs instances between 3 and 7 of experiment 1 in the GPU number 1

## CUSTOM MODELS DEFINITION
To define a model follow the instructions written in define_models.EXAMPLE_model()


## PREPROCESSING
Unfortunately, the preprocessing needs to be customized for every new dataset. In order to be compatible con the rest of the API, any proprocessing script has to output 2 dictionaries: 1 containing the predictors and 1 containing the target. The keys of these dicts has to be the 'foldable items', that is the criterion that you want to use to divide train-va-test, for example the different actors in a speech dataset. So every key should contain all data from one single actor.

Example:
```python
predictors_dict['1': matrix with all spectra of actor 1,
                '2': matrix with all spectra of actor 2]
target_dict['1': matrix with all labels of actor 1,
            '2': matrix with all labels of actor 1]
```
