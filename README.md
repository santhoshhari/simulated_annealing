# Hyperparameter Tuning using Simulated Annealing

Machine learning models with high number of hyperparameters usually perform better on a wide variety of datasets owing to the flexibility they offer. One such model that is used often is XGBoost. XGBoost has more than 15 parameters and the choice of these hyperparameter values dictate the performance of the model (apart from the features). Thus, tuning hyperparameters is important to achieve maximum effectiveness from the model. 

Approaches often employed to tune the hyperparameters are:
1. **Grid Search**: This method does an exhaustive search in the hyperparmeter space. This exhaustive search results in high run time, owning to the total number of combinations that emerge in cases with many hyperparameters and possible values they can take.
2. **Random Search**: This method draws random samples from the hyper parameter domain (continuous or discrete). It solves the problem of run time by exploring fewer combinations by but may not always yeild a result as good as grid search. 
3. **Sequential Grid Search**: This method is a sequential application of Grid Search where the hyper parameters (one or a subset) are tuned sequentially. This avoids high number of combinations but fails to capture  the interactions between hyperparameters.

To overcome the discussed drawbacks, one can use *Simulated Annealing* to find the optimal hyper parameter values.

## Simulated Annealing
Simulated Annealing can be used to find close to optimal solution in a discrete search space with large number of possible solutions (combination of hyperparameters). It is useful for combinatorial optimization problems defined by complex objective functions (model evaluation metrics).

> Simulated Annealing is a probabilistic technique for approximating global optimum of a given function.

The name of this approach is inspired from metallurgy technique of heating and controlled cooling of materials called `annealing`.

> This notion of controlled cooling implemented in the simulated annealing algorithm is interpreted as a slow decrease in the probability of accepting worse solutions as the solution space is explored.

> At each step, the simulated annealing heuristic considers some neighboring state s* of the current state s, and probabilistically decides between moving the system to state s* or staying in-state s. These probabilities ultimately lead the system to move to states of lower energy.

Simulated Annealing applied to hyper parameter tuning consists of following steps:

1. Randomly choose a value for all hyperparameters and treat it as current state and evaluate model performance
2. Alter the current state by randomly updating value of one hyperparameter by selecting a value in the immediate neighborhood (randomly) to get neighbouring state
3. If the combination is already visited, repeat step 2 until a new combination is generated
4. Evaluate model performance on the neighbouring state
5. Compare the model performance of neighbouting state to the current state and decide whether to accept the neighbouring state as current state or reject it based on some criteria (explained below). 
6. Based on the result of step 5, repeat steps 2 though 5

**Acceptance Criteria:**
- If the performance of neighbouring state is better than current state - Accept
- If the performance of neighbouring state is worse than current state - Accept with probability <img src="https://latex.codecogs.com/gif.latex?e^{-\beta&space;\Delta&space;f/T}" title="e^{-\beta \Delta f/T}" /> where,
    - <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> is a constant
    - T is current temperature
    - <img src="https://latex.codecogs.com/gif.latex?\Delta&space;f" title="\Delta f" /> is the performance difference between the current state and the neighbouring state
    
**Annealing Parameters:**

Simulated annealing algorithms takes in four parameters and the effectiveness of the algorithm depends on the choice of thse parameters.

1. <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> - normalizing constant

> Choice of <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> depends on the expected variation in the performance measure over the search space. This can be chosen by playing around with [this](https://github.com/santhoshhari/simulated_annealing/blob/master/simulated_annealing_parameters.xlsx) Excel workbook. If the chosen value of beta is too high, i.e., probability of rejecting a set of parameters is too low in later iterations, you may end up in an infinite loop.

2. <img src="https://latex.codecogs.com/gif.latex?T_0" title="T_0" /> - initial temperature

> A good rule of thumb is that your initial temperature <img src="https://latex.codecogs.com/gif.latex?T_0" title="T_0" /> should be set to accept roughly 98% of the moves and that the final temperature should be low enough that the solution does not improve much, if at all

3. <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> - Factor by which temperature is scaled after n iterations.

> Lower values of <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> restrict the search space at a faster rate than higher values.  0.85 can be chosen by default.

4. n - number of iterations after which temperature is altered (after every n steps T is updated as T * <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" />

> The value of n doesn't affect the results and can be chosen between 5 - 10.

## Usage

A version of simulated annealing has been implemented and available in the `simmulated_annealing.py`. It can be downloaded and imported using the following command
`from simulated_annealing import *`

annealing_example notebook shows how to use the current implementation. You have to define train_model and parameter dictionaries before calling the simulate_annealing function.
