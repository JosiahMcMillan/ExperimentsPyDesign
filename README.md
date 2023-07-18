
# Background
The purpose of this package is merely to extend what is available in the experimental design space in Python, right now a lot of functionality is in R or on JMP and some of the experimental designs that I was exposed to in the course of my studies were not available in Python. This package is hopefully useful for filling those gaps in the space, letting users design their own experiments for their home projects or in whatever setting they please. Another nice Design of Experiments package that I was inspired by is the [pyDOE2 package](https://github.com/clicumu/pyDOE2) which has a lot of nice functionality.

Please raise an issue or message me directly if you are running into any errors. I intend on expanding the functionality of this package and possibly changing some of the function calls to be more consistent with the style of other packages, see the [TODO section](#todo).


# Usage:
The input to these functions generally consists of:

variables: A list of variable names for the experiment. This could be something like ['temperature', 'pressure', 'time'] for an experiment with three factors.

levels: A list of lists where each inner list represents the different levels of the corresponding variable in the variables list.

## Example:

input:
```
design <- generate_taguchi_design(['temperature', 'pressure'], [[100, 200], [1, 2]])

print(design)
```


output:
|   temperature |   pressure |
|--------------:|-----------:|
|           100 |          1 |
|           100 |          2 |
|           200 |          1 |
|           200 |          2 |

## Function descriptions:

#### `generate_taguchi_design(variables, levels)`
Generates a Taguchi design. This function takes a list of variables and their corresponding levels as input and produces a design matrix following the principles of the Taguchi method. The design matrix represents the combinations of levels for each variable.

#### `generate_rsm_design(variables, ranges)`
Generates a response surface methodology (RSM) design. This function takes a list of variables and their corresponding ranges as input and generates a design matrix that represents the combinations of levels within the specified ranges. RSM designs are commonly used for fitting response surface models to study the relationships between variables and responses.

#### `generate_plackett_burman_design(variables, factors)`
Generates a Plackett-Burman design. This function takes a list of variables and the number of factors as input and produces a design matrix that follows the Plackett-Burman experimental design technique. Plackett-Burman designs are widely used for screening experiments to identify the most influential factors affecting a process or system.

#### `generate_simplex_lattice_design(n_variables, n_levels)`
Generates a Simplex Lattice design. This function generates a design matrix for a Simplex Lattice design, which is a specialized design commonly used in mixture experiments. It ensures that the design points lie on a simplex, a geometric shape representing the composition proportions of the components in a mixture, and ensures a balanced allocation of design points across the mixture space.

#### `generate_ccd_design(variables, ranges)`
Generates a central composite design (CCD). This function takes a list of variables and their corresponding ranges as input and generates a design matrix following the principles of a central composite design. CCDs are often used in response surface methodology to study the relationships between variables and responses, including both linear and quadratic effects.

#### `generate_box_behnken_design(variables, ranges)`
Generates a Box-Behnken design. This function takes a list of variables and their corresponding ranges as input and produces a design matrix following the Box-Behnken experimental design technique. Box-Behnken designs are used for response surface modeling and aim to reduce the number of experimental runs required while capturing the quadratic response behavior.

#### `generate_augmented_design(variables, factors, interactions)`
Generates an augmented design. This function takes a list of variables, the number of factors, and the number of desired interactions as input and produces an augmented design matrix. Augmented designs are used to augment existing designs, such as Plackett-Burman or fractional factorial designs, by adding additional runs to enhance the estimation of interaction effects.

#### `generate_split_plot_design(variables, plot_factors, treatment_factors)`
Generates a split-plot design. This function takes a list of variables, the number of plot factors, and the number of treatment factors as input and produces a split-plot design matrix. Split-plot designs are used when the experimental units are divided into subunits, with different factors applied at different levels of the subunits and main plots.

#### `generate_graeco_latin_square_design(variables, levels)`
Generates a Graeco-Latin square design. This function takes a list of variables and their corresponding levels as input and generates a design matrix following the Graeco-Latin square design technique. Graeco-Latin square designs are used to efficiently allocate treatments to experimental units, ensuring that each treatment appears once in each row and column, minimizing potential confounding effects.

#### `generate_youden_square_design(variables, levels)`
Generates a Youden square design. This function takes a list of variables and their corresponding levels as input and produces a design matrix following the Youden square design technique. Youden square designs are used in experiments where multiple factors are being studied simultaneously, with the goal of estimating main effects and interactions without confounding.

#### `generate_completely_randomized_design(variables, levels, n_points)`
Generates a completely randomized design. This function takes a list of variables, their corresponding levels, and the number of desired points as input and produces a completely randomized design matrix. Completely randomized designs are used when the experimental units can be randomly assigned to different factor combinations, ensuring each combination has an equal chance of being tested.

#### `generate_randomized_block_design(variables, levels, n_blocks)`
Generates a randomized block design. This function takes a list of variables, their corresponding levels, and the number of desired blocks as input and produces a randomized block design matrix. Randomized block designs are used when the experimental units can be grouped into blocks, and treatments are randomly assigned within each block to account for potential blocking effects.

#### `generate_factorial_design_with_nested_factors(variables, levels, nested_variables, nested_levels)`
Generates a factorial design with nested factors. This function takes a list of variables, their corresponding levels, a list of nested variables, and their levels as input and produces a factorial design matrix with nested factors. This design allows for studying both the main effects and the interactions between factors and nested factors.

#### `generate_split_plot_fractional_factorial_design(variables, plot_factors, treatment_factors, factors)`
Generates a split-plot fractional factorial design. This function takes a list of variables, the number of plot factors, the number of treatment factors, and the number of factors as input and produces a split-plot fractional factorial design matrix. This design allows for studying a fraction of the total number of factors while considering the split-plot structure.

#### `generate_starting_design(n, k)`
Generates a starting design for the coordinate exchange algorithm. This function takes the number of rows `n` and the number of columns `k` as input and generates a starting design matrix for the coordinate exchange algorithm. The starting design serves as an initial configuration for the algorithm to iteratively improve upon.

#### `construct_split_plot_design(num_runs, a_priori_model, num_factors, factor_types, constraints, num_starting_designs, num_rows, num_columns, var_ratios)`
Constructs a split-plot design. This function constructs a split-plot design matrix based on the specified parameters. It takes into account the number of runs, the a priori model, the number of factors, the types of factors, constraints, the number of starting designs, the number of rows, the number of columns, and the variation ratios. The resulting design matrix reflects the split-plot structure and the relationships between the specified factors.

#### `coordinate_exchange_algorithm(n, k, beta, max_iter=1000, num_starts=1000)`
Performs the coordinate exchange algorithm. This algorithm aims to optimize the design matrix by iteratively exchanging values within the matrix to maximize a specific criterion. It takes the number of rows `n`, the number of columns `k`, the optimization criterion `beta`, and optional parameters such as the maximum number of iterations and the number of starting designs. The algorithm returns an optimized design matrix.

#### `coordinate_exchange_algorithm_augmentation(n2, k, beta, X1, max_iter=1000, num_starts=1000)`
Performs the coordinate exchange algorithm with augmentation. This algorithm extends the coordinate exchange algorithm by including an augmentation process. It takes the number of rows `n2`, the number of columns `k`, the optimization criterion `beta`, an initial design matrix `X1`, and optional parameters such as the maximum number of iterations and the number of starting designs. The algorithm returns an optimized design matrix that incorporates the augmentation process.




# TODO
- Make some additional input checks where required
- Probably need to implement lack-of-fit test
- Need to implement some other plots for checking the models beyond what we have
- Additional work on the simulation studies and Bayesian design
- Adaptive designs
- Provide more info + examples of the efficiency measures + how the coordinate exchange algorithm works
