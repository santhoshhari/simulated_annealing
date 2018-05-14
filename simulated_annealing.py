import pandas as pd
import numpy as np
from collections import OrderedDict
from random import random

def choose_params(tune_dic, curr_params=None):
    """
    Function to choose parameters for next iteration
    
    Inputs:
    tune_dic - Dict of Hyperparameter search space
    curr_params - Dict of current hyperparameters
    
    Output:
    Dictionary of parameters
    """
    
    if curr_params:
        next_params = curr_params.copy()
        param_to_update = np.random.choice(list(tune_dic.keys()))
        param_vals = tune_dic[param_to_update]
        curr_index = param_vals.index(curr_params[param_to_update])
        if curr_index == 0:
            next_params[param_to_update] = param_vals[1]
        elif curr_index == len(param_vals) - 1:
            next_params[param_to_update] = param_vals[curr_index - 1]
        else:
            next_params[param_to_update] = \
                param_vals[curr_index + np.random.choice([-1,1])]
    else:
        next_params = dict()
        for k, v in tune_dic.items():
            next_params[k] = np.random.choice(v)

    return next_params


def simulate_annealing(fn_train,
                       tune_dic,
                       const_param,
                       X_train,
                       X_valid,
                       Y_train=None,
                       Y_valid=None,
                       maxiters=100,
                       alpha=0.85,
                       beta=1.3,
                       T=0.40,
                       update_iters=5,
                       fn_train_args=None):
    """
    Function to perform hyperparameter search using simulated annealing
    
    Inputs:
    fn_train - Function to train the model (Should return model and metric value)
    tune_dic - Dictionary of Hyperparameter search space
    maxiters - Number of iterations to perform the parameter search
    alpha - factor to reduce temperature
    beta - constant in probability estimate
    T - Initial temperature
    update_iters - # of iterations required to update temperature
    fn_train_args - Dictionary of non-trivial parameters (Everything except curr_params, param, X_train, X_valid)
    
    Output:
    Dataframe of the parameters explored and corresponding model performance
    """

    columns = [*tune_dic.keys()] + ['Metric', 'Best Metric']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_metric = -1.
    prev_metric = -1.
    prev_params = None
    best_params = dict()
    weights = list(map(lambda x: 10**x, list(range(len(tune_dic)))))
    hash_values = set()

    for i in range(maxiters):
        print('Starting Iteration {}'.format(i))
        while True:
            curr_params = choose_params(tune_dic, prev_params)
            indices = [tune_dic[k].index(v) for k, v in curr_params.items()]
            hash_val = sum([i * j for (i, j) in zip(weights, indices)])
            if hash_val in hash_values:
                print('Combination revisited')
            else:
                hash_values.add(hash_val)
                break

        if fn_train_args:
            model, metric = fn_train(curr_params, const_param, X_train, X_valid, **fn_train_args)
        else:
            model, metric = fn_train(curr_params, const_param, X_train, X_valid)

        if metric > prev_metric:
            print('Local Improvement in metric from {:8.4f} to {:8.4f} - parameters accepted'\
                  .format(prev_metric, metric))
            prev_params = curr_params.copy()
            prev_metric = metric

            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} - best parameters updated'\
                  .format(best_metric, metric))
                best_metric = metric
                best_params = curr_params.copy()
        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            threshold = np.exp(beta * diff / T)
            if rnd < threshold:
                print(
                    """No Improvement but parameters accepted. Metric change: {:8.4f} 
                threshold: {:6.4f} random number: {:6.4f}
                """.format(diff, threshold, rnd))
                prev_metric = metric
                prev_params = curr_params
            else:
                print(
                    """No Improvement and parameters rejected. Metric change: {:8.4f} 
                threshold: {:6.4f} random number: {:6.4f}
                """.format(diff, threshold, rnd))

        results.loc[i, list(curr_params.keys())] = list(curr_params.values())
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric

        if i % update_iters == 0: T = alpha * T

    return results
