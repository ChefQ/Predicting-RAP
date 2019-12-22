import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
from keras import layers
from keras import models
import numpy as np 
import pandas as pd
"""
Builds a nn regression modelall_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories
"""
def build_model(units,layerz,metric,shape):
    model = models.Sequential()
    model.add(layers.Dense(units, activation = 'relu', 
                           input_shape = (shape,)))
    for layer in range(layerz):
        model.add(layers.Dense(units, activation = 'relu'))
    model.add(layers.Dense(9, activation = 'softmax')) #linear layer
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics =[metric] )
    return model

def to_one_hot(labels, dimension = 9):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, int(label)-1] = 1
    return results

def a_p_score(df : pd.core.frame.DataFrame, DGM : float, G : str, column : str):
    num_rows = df.shape[0]

    
    ID = G
    
    IDs = set(df[ID].tolist())

    df = df.sort_values(by=ID)
    scores = np.empty((0,1)) # digging actor or partner
    
    for I in IDs:
        person_df = df.loc[df[ID] == I,column]
        mean = person_df.mean()
        num_dates = person_df.shape[0]

        score = mean * np.ones((num_dates,1))
        scores = np.concatenate((scores,score), axis = 0 )
                 

    a_p_scores =  scores - DGM 
    
    return a_p_scores

"""
params: units, layerz, metric,num_epochs
returns : mae, loss, validation loss and validition mae histories for each fold
          tuple( all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories)
"""
def validaton(hyper_parameters,shape,data):
    k = 4
    
    units, layerz = hyper_parameters['units'], hyper_parameters['layerz']
    metric,num_epochs = hyper_parameters['metric'], hyper_parameters['num_epochs']
    train_data,train_targets = data['train'],data['target']
    
    num_val_samples = len(train_data) // k
    all_val_mae_histories = []
    all_val_loss_histories = []
    all_loss_histories = []
    all_mae_histories = []
    
    print("val_samples: " + str(num_val_samples))
    for i in range(k):
        print('\tprocessing fold #', i)
        start = i*num_val_samples
        stop = (i + 1) * num_val_samples
        val_data = train_data[start:stop]
        val_target = train_targets[start:stop]

        partial_train_data = np.concatenate( (train_data[:start], train_data[stop:]) ,axis = 0)
        partial_train_target = np.concatenate( (train_targets[:start],train_targets[stop:]), axis = 0) 

        model = build_model(units, layerz, metric,shape)
        history = model.fit(partial_train_data, partial_train_target, epochs = num_epochs, batch_size = 1, verbose = 0,
                            validation_data = (val_data, val_target) )


        val_mae_history = history.history['val_mean_absolute_error']
        val_loss_history = history.history['val_loss']
        loss_history = history.history['loss']
        mae_history = history.history['mean_absolute_error']

        all_val_mae_histories.append(val_mae_history)  
        all_val_loss_histories.append(val_loss_history)
        all_loss_histories.append(loss_history)
        all_mae_histories.append(mae_history)

    return all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories

"""
returns the histories of all the P * K folds
"""
def iterated_validation(num_iterations,hyper_parameters,shape,data):
    all_val_mae_histories = []
    all_val_loss_histories = []
    all_loss_histories = []
    all_mae_histories = []
    train_data,train_targets = data['train'],data['target']
    for i in range(num_iterations):
        print("iteration: "+str(i+1))
            # shuffle training data
        rows = np.arange(train_targets.size)
        indexes = shuffle(rows)
        train_data = train_data[indexes]
        train_targets = train_targets[indexes]
        print("\tStarting indexes for training " +str(indexes[0:5]))
        
        histories = validaton(hyper_parameters,shape,data)
        all_val_mae_histories.extend(histories[0])
        all_val_loss_histories.extend(histories[1])
        all_loss_histories.extend(histories[2])
        all_mae_histories.extend(histories[3])
        
    return all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories

"""
Returns the mean for each elements returned in by the 
iterated_validation or validation function
"""
def average_folds(all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories):
    averages = []
    num_epochs = len(all_val_mae_histories[0])
    averages.append([np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)])
    averages.append([np.mean([x[i] for x in all_val_mae_histories]) for i in range(num_epochs)])
    averages.append([np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)])
    averages.append([np.mean([x[i] for x in all_val_loss_histories]) for i in range(num_epochs)])
    
    return averages

"""
Creates and plots 4 subplots. That represents 
the loss and Mae for the training and validation set
"""
def plots(histories):
    epoch = range(1,len(histories['val_mean_absolute_error'])+1) 
    f, axes = plt.subplots(2, 2, figsize=(12,12))
    axes = axes.reshape((2,))
  
    axes[0].plot(epoch,histories['val_mean_absolute_error'], label='Training')
    axes[1].plot(epoch,histories['mean_absolute_error'], 'r', label='Validation')
    axes[1].legend()
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")

    axes[2].set_ylabel("Loss")
    
def plot_features(features, names, smooth=0):
    val_mae_index = 1
    val_loss_index = 3
    epochs = range(1,len(features[names[0]][val_mae_index])+1)
    color = ["b","r","g","k"]
    plt.figure(figsize=(12,12))
    plt.title('overfitting with different size features')
    plt.xlabel('Epochs')
    plt.ylabel('Validation mae')
    for i,name in enumerate(names):
        if smooth != 0:
            plt.plot(epochs, smooth_curve(features[names[i]][val_mae_index]), color[i], label=name) 
        else:
            plt.plot(epochs, features[names[i]][val_mae_index], color[i], label=name) 
        plt.legend()
        print(name + " had a " + str(np.min(features[names[i]][val_mae_index])) +" MAE value")
        print(name + " had a " + str(np.min(features[names[i]][val_loss_index])) +" Loss value")
        print()
    plt.show()
    

    
"""
Manipulates the data such that it looks smoother when you plot it
"""
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point*(1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# types of features selections criteria i used

criteria = ["CR", "MI","DI","DA"] 
"""

"""
def bestKFeautues(criteria,K,train_datas, train_target):
    features = {}
    for i in range(len(criteria)):
        print("\t\t\t\t\t\t\t "+str(criteria[i]))
        hyper_parameters = dict(units = 32, layerz = 2, metric ='mae', num_epochs = 100)
        shape = K
        data = dict(train= train_datas[i], target = train_target)
        all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories = iterated_validation(1, hyper_parameters,shape,data)

        # find the averages
        averages = average_folds(all_val_mae_histories, all_val_loss_histories, all_loss_histories, all_mae_histories)
        #import pdb; pdb.set_trace()
        # set appriopriate feature criteria to averages
        features[criteria[i]] = (averages[0], averages[1], averages[2],averages[3])
    return features

def normalize(X, indices):
    normalized_train_datas = []
    normalized_test_datas = []
    mean = X[0:300,:].mean(axis = 0)
    normalized_X = X.copy()
    normalized_X[0:300,:] -= mean
    std = X.std(axis = 0)
    normalized_X /= std

    for i in range(len(indices)): # should change this the test_data has information about the whole dataset
        normalized_train_datas.append(normalized_X[0:300,indices[i]])
        normalized_test_datas.append(normalized_X[300:,indices[i]])
    return normalized_train_datas