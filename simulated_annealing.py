import pandas as pd
import numpy as np
from collections import OrderedDict
from random import random


# Sample train_model function #
###############################

# def train_model(curr_params, param, Xtrain, Xvalid, Ytrain,
#                 Yvalid):
#     """
#     Train the model with given set of hyperparameters
#     curr_params - Dict of hyperparameters and chosen values
#     param - Dict of hyperparameters that are kept constant
#     Xtrain - Train Data
#     Xvalid - Validation Data
#     Ytrain - Train labels
#     Yvalid - Validaion labels
#     """
#     params_copy = param.copy()
#     params_copy.update(curr_params)
#     model = XGBClassifier(**params_copy)
#     model.fit(Xtrain, Ytrain)
#     preds = model.predict(Xvalid)
#     metric_val = f1_score(Yvalid, preds) # Any metric can be used
#     return model, metric_val


def choose_params(param_dict, curr_params=None):
    """
    Function to choose parameters for next iteration
    Inputs:
    param_dict - Ordered dictionary of hyperparameter search space
    curr_params - Dict of current hyperparameters
    Output:
    Dictionary of parameters
    """
    if curr_params:
        next_params = curr_params.copy()
        param_to_update = np.random.choice(list(param_dict.keys()))
        param_vals = param_dict[param_to_update]
        curr_index = param_vals.index(curr_params[param_to_update])
        if curr_index == 0:
            next_params[param_to_update] = param_vals[1]
        elif curr_index == len(param_vals) - 1:
            next_params[param_to_update] = param_vals[curr_index - 1]
        else:
            next_params[param_to_update] = \
                param_vals[curr_index + np.random.choice([-1, 1])]
    else:
        next_params = dict()
        for k, v in param_dict.items():
            next_params[k] = np.random.choice(v)

    return next_params


def simulate_annealing(param_dict,
                       const_param,
                       X_train,
                       X_valid,
                       Y_train,
                       Y_valid,
                       fn_train,
                       maxiters=100,
                       alpha=0.85,
                       beta=1.3,
                       T_0=0.40,
                       update_iters=5):
    """
    Function to perform hyperparameter search using simulated annealing
    Inputs:
    param_dict - Ordered dictionary of Hyperparameter search space
    const_param - Static parameters of the model
    Xtrain - Train Data
    Xvalid - Validation Data
    Ytrain - Train labels
    Yvalid - Validaion labels
    fn_train - Function to train the model
        (Should return model and metric value as tuple, sample commented above)
    maxiters - Number of iterations to perform the parameter search
    alpha - factor to reduce temperature
    beta - constant in probability estimate
    T_0 - Initial temperature
    update_iters - # of iterations required to update temperature
    Output:
    Dataframe of the parameters explored and corresponding model performance
    """
    columns = [*param_dict.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = -1.
    prev_metric = -1.
    prev_params = None
    best_params = dict()
    weights = list(map(lambda x: 10**x, list(range(len(param_dict)))))
    hash_values = set()
    T = T_0

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))
        while True:
            curr_params = choose_params(param_dict, prev_params)
            indices = [param_dict[k].index(v) for k, v in curr_params.items()]
            hash_val = sum([i * j for (i, j) in zip(weights, indices)])
            if hash_val in hash_values:
                print('Combination revisited')
            else:
                hash_values.add(hash_val)
                break

        model, metric = fn_train(curr_params, const_param, X_train,
                                 X_valid, Y_train, Y_valid)

        if metric > prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - parameters accepted')
            prev_params = curr_params.copy()
            prev_metric = metric

            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) +
                      ' - best parameters updated')
                best_metric = metric
                best_params = curr_params.copy()
                best_model = model
        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            threshold = np.exp(beta * diff / T)
            if rnd < threshold:
                print('No Improvement but parameters accepted. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))
                prev_metric = metric
                prev_params = curr_params
            else:
                print('No Improvement and parameters rejected. Metric change' +
                      ': {:8.4f} threshold: {:6.4f} random number: {:6.4f}'
                      .format(diff, threshold, rnd))

        results.loc[i, list(curr_params.keys())] = list(curr_params.values())
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric

        if i % update_iters == 0:
            T = alpha * T

    return results, best_model
