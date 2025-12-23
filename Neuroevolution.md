# Neuroevolution: harnessing Creativig in AI Agent Design

This is a summary of the book [Neuroevolution: harnessing Creativig in AI Agent Design](https://neuroevolutionbook.com/).


# Chapter 2 The Basics.
## 2.1. Evolutionary Algorithm 
The basic solver loop.
```
solver = EvolutionAlgorithm()
while True:
    # Ask the EA to give us a set of candidate solutions.
    solutions = solver.ask()
    # Create an array to hold the fitness results.
    fitness_list = np.zeros(solver.popsize)
    # Evaluate the fitness for each given solution.
    for i in range(solver.popsize): 
        fitness_list[i]= evaluate(solutions[i])
        # Give list of fitness results back to EA.
        solver.tell(fitness_list)
        # Get best parameter, fitness from EA.
        best_solution, best_fitness = solver.result()
        if best_fitness > MY_REQUIRED_FITNESS:
            break
```
## 2.1.1 Representation
 - genotype: internal data structure used by the algorithm to represent a candidate solution - typically a string, vector, tree, or graph structure thatis subject to variation and selection
 - Phenotype: external manifestation of this solution in the context of the problem domain.

## 2.1.2 population-based search
- the population refers to the set of individuals maintained and evolved over successive generations.
    - Smaller populations tend to converge quickly butrisk premature convergence due to insuﬃcient diversity. 
    - Larger populations maintain broader coverage of the search space but can slow down convergence and increase resourcedemand

## 2.1.3 selection
From generation to generation. 
- high selection: reduce genetic diversity and may cause premature convergence.
- low selection: weaker individuals a chance to reproduce, which slows convergence but promotes diversity and broader exploration of the searchspace. 

## 2.1.4 variation operators
From generation to generation.
- mutations: alters individuals randomly.
- crossovers: combines traits from two or more parents.

## 2.2. Types of Evolutionary Algorithms

### 2.2.1. Genetic Algorithm (GA)
- Mostly it's about cross-over

### 2.2.2. Evolution Strategy (ES)
- Mostly it's about mutations

### 2.2.3. Covariance-Matrix Adaptation Evolution Strategy (CMA-ES)
- Mostly it's about better mutations (adapt variance as generation goes by)

### 2.2.4. OpenAI Evolution Strategy
- 

### 2.2.5. Multiobjective Evolutionary Algorithms

## CODE Exercise.

### base algorithm

Let's think about the task of finding a global mininmum (or maximum) using Evolution stareteiges (ES) and genetic algorithm (GA)

We first provide the basic of agent. It have ask and tell function. The function `ask` return a population of a soluions of the next generation. Function `tell` updates internal state based on fitness scores that it gets as argument.

```
from typing import List, Union


class BaseAlgo(object):
    """Interface definition for all ES/GA algorithms."""

    pop_size: int       # Size of the population.
    num_params: int     # num_params=2 here because target functions are in 2D, num_params is a dimension of soluiton.

    def ask(self) -> np.ndarray:
        """Return a population of solutions for the next generation.

        Returns:
          An array of size (pop_size, num_params).
        """
        raise NotImplementedError()

    def tell(self, fitness: Union[np.ndarray, List]) -> None:
        """Update the internal state based on the fitness scores.

        Arguments:
          fitness - An array of size pop_size, representing the fitness score
                    for each of the individual in the population.
        """
        raise NotImplementedError()
```


### Simple ES

Based on above template, let's first implement simple evolutionary strategy (simple ES). We keep our attention to find a global minimum $(x^*,y^*)$ of function $f$.

Let's say at generation $m$, 

The `tell` function would take the the fitness score of the total $N$ number of points $\{(x_i,y_i)\}_{i \in [N]}$ as $\{ f_{i}\}_{i \in [N]}$ and update the internal state $(s^{(m)},t^{(m)})$ as the point that returns the minimum of $\{ f_{i}\}_{i \in [N]}$, i.e. update internal state as $(s^{(m)},t^{(m)}) = (x_n,y_n)$ where $n = \argmin_{i \in [N]} f_i$.

Then the `ask` function would samples a population of $M$ generation as total $N$ number from updated $(s^{(m)},t^{(m)})$. Specifically, total $N$ number are sampled from gaussian distribution where mean is $(s^{(m)},t^{(m)})$ and the std is fixed number $\sigma$. 

$$(x_i,y_i) \sim \mathcal{N} ((s^{(m)},t^{(m)}), \sigma)$$

We define this as "Simple ES" algorithm 

```
class SimpleES(BaseAlgo):
    """Your should implement this class."""

    def __init__(self,
                 pop_size,
                 num_params,
                 init_x,
                 stdev,
                 seed):
        """Initialize the internal states.

        Arguments:
          pop_size - Population size.
          num_params - Number of parameters to optimize.
          init_x - Initial guess of the solution.
          stdev - Standard deviation used for population sampling.
          seed - Random seed.
        """
        self.pop_size=pop_size
        self.num_params=num_params
        self.mean = init_x
        self.stdev = stdev

        self.rng = np.random.default_rng(seed=seed)
        self.sample = np.zeros((self.pop_size,self.num_params))

    def ask(self) -> np.ndarray:
        """Return a population of solutions for the next generation.

        Returns:
          An array of size (pop_size, num_params).
        """
        self.samples = self.rng.normal(loc=self.mean, scale=self.stdev, size=(self.pop_size, self.num_params))

        return self.samples

    def tell(self, fitness: Union[np.ndarray, List]) -> None:
        """Update the internal state based on the fitness scores.

        Arguments:
          fitness - An array of size pop_size, representing the fitness score
                    for each of the individual in the population.
        """
        ix = np.argmin(fitness)
        self.mean = self.sample[i,:]
```

### Simple GA

Now, we implement simple genetic algorithm. Note that genetic algorithm is composed of mainly two parts 
 - for given $N$ number of individuals at generation $m$, keep top $n <N$ individuals to next generation $m+1$. 
 - for rest of $N-n$ individuals, pick 2 individuals and do crossover (single-point, two-point, uniform) for $ N-n$ times and move to next generation $m+1$.
    - how to pick 2 individuals? Roulette wheel, tournament, rank-based, etc.

```
class SimpleGA(BaseAlgo):
    """Your should implement this class."""

    def __init__(self,
                 pop_size,
                 num_params,
                 init_x,
                 stdev,
                 elite_ratio,
                 seed):
        """Initialize the internal states.

        Arguments:
          pop_size - Population size.
          num_params - Number of parameters to optimize.
          init_x - Initial guess of the solution.
          stdev - Standard deviation used for population sampling.
          elite_ratio - Ratio of elites to keep.
          seed - Random seed.
        """
        self.pop_size=pop_size
        self.num_params=num_params
        self.elite_size = int(self.pop_size * elite_ratio)

        self.rng = np.random.default_rng(seed=seed)
        self.population = self.rng.normal(loc=init_x, scale=stdev, size=(self.pop_size, self.num_params))


    def ask(self) -> np.ndarray:
        """Return a population of solutions for the next generation.

        Returns:
          An array of size (pop_size, num_params).
        """
        return self.population

    def tell(self, fitness: Union[np.ndarray, List]) -> None:
        """Update the internal state based on the fitness scores.

        Arguments:
          fitness - An array of size pop_size, representing the fitness score
                    for each of the individual in the population.
        """
        # [1] first keep the elite 
        fitness = np.array(fitness)
        # Sort indices by descending fitness (lower fitness is better)
        elite_idx = np.argsort(fitness.squeeze())[:self.elite_size]
        # Save elites
        elites = [self.population[idx,:] for idx in elite_idx]
        
        
        # [2] Then we mutate the rest of them 
        # Calculate selection probabilities (normalize fitness for prob)
        fit = fitness - np.min(fitness)
        probs = fit / np.sum(fit) if np.sum(fit) > 0 else np.ones_like(fitness) / len(fitness)
        # Sample nonelite children for remainder of population

        num_children = self.pop_size - self.elite_size
        num_parents = 2 * num_children
        selected_idx = self.rng.choice(
            len(self.population), size=num_parents, replace=True, p=probs.squeeze())
        children = []
        for i,idx in enumerate(selected_idx):
            if i % 2 == 0 : 
              continue
            # Deep copy to avoid mutation affecting original parent
            import copy
            parent1 = copy.deepcopy(self.population[idx-1])
            parent2 = copy.deepcopy(self.population[idx])
            # Generate a random boolean mask of the same shape as the parents
            mask = np.random.choice([True, False], size=parent1.shape)
            offspring = np.where(mask, parent1, parent2)
            children.append(offspring)
        new_population = elites + children
        self.population= np.array(new_population)

        

```

