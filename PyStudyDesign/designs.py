import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product
# from .bayesian import create_model, bayesian_optimal_design
from scipy.special import comb
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from numpy.linalg import inv, eigvals


def save_design_as_text(design, filename):
    design.to_csv(filename, index=False)
    

def check_input(variables, levels, n_points=None):
    if not isinstance(variables, list):
        raise TypeError("variables should be a list")
    if not isinstance(levels, list):
        raise TypeError("levels should be a list")
    if len(variables) != len(levels):
        raise ValueError("variables and levels should have the same length")
    if n_points is not None and not isinstance(n_points, int):
        raise TypeError("n_points should be an integer")


def generate_taguchi_design(variables, levels):
    check_input(variables, levels)
    matrix = np.array(list(product(*levels)))
    return pd.DataFrame(matrix, columns=variables)


def generate_rsm_design(variables, ranges):
    check_input(variables, ranges)
    matrix = np.array(list(product(*ranges)))
    return pd.DataFrame(matrix, columns=variables)


def generate_plackett_burman_design(variables, factors):
    if not isinstance(factors, int):
        raise TypeError("factors should be an integer")
    check_input(variables, [None] * len(variables))

    n = len(variables)
    m = factors if factors < n else n - 1
    design = norm.rvs(size=(2 ** m, n))
    design = np.where(design >= 0, 1, -1)
    return pd.DataFrame(design, columns=variables)


def generate_simplex_lattice_design(n_variables, n_levels):
    """Generates a Simplex Lattice Design."""
    check_input(n_variables * [None], n_levels * [None])

    n_points = comb(n_levels + n_variables - 1, n_variables - 1, exact=True)
    design = np.zeros((n_points, n_variables))

    row = 0
    for level_values in product(range(n_levels), repeat=n_variables):
        if sum(level_values) == (n_levels - 1):
            design[row] = [lv / (n_levels - 1) for lv in level_values]
            row += 1

    return pd.DataFrame(design, columns=[f"Variable {i+1}" for i in range(n_variables)])


def generate_ccd_design(variables, ranges):
    # Input validation
    check_input(variables, ranges)
    matrix = np.array(list(product(*ranges)))
    center_points = np.zeros((1, len(variables)))
    center_points[0, :] = np.mean(matrix, axis=0)
    design = np.concatenate((matrix, center_points))
    return pd.DataFrame(design, columns=variables)


def generate_box_behnken_design(variables, ranges):
    # Input validation
    check_input(variables, ranges)

    n = len(variables)
    center_points = np.zeros((1, n))
    for i in range(n):
        center_points[0, i] = (ranges[i][0] + ranges[i][2]) / 2
    matrix = np.zeros((n + 2 * (n * (n - 1) // 2), n))
    matrix[:n, :] = np.array(list(product(*ranges)))
    idx = n
    for i in range(n):
        for j in range(i + 1, n):
            matrix[idx, i] = ranges[i][0]
            matrix[idx, j] = ranges[j][2]
            idx += 1
            matrix[idx, i] = ranges[i][2]
            matrix[idx, j] = ranges[j][0]
            idx += 1
    design = np.concatenate((matrix, center_points))
    return pd.DataFrame(design, columns=variables)


def generate_augmented_design(variables, factors, interactions):
    # Input validation
    check_input(variables, [], factors=factors)
    assert isinstance(interactions, int), "interactions should be an integer"

    n = len(variables)
    m = factors if factors < n else n - 1
    design = generate_plackett_burman_design(variables, m).to_numpy()
    interactions_design = np.eye(interactions, n)
    augmented_design = np.concatenate((design, interactions_design))
    return pd.DataFrame(augmented_design, columns=variables)


def generate_split_plot_design(variables, plot_factors, treatment_factors):
    # Input validation
    check_input(variables, [], plot_factors=plot_factors, treatment_factors=treatment_factors)

    plot_design = generate_plackett_burman_design(variables[:plot_factors], plot_factors)
    treatment_design = generate_plackett_burman_design(variables[plot_factors:], treatment_factors)

    design = np.repeat(plot_design.values, len(treatment_design), axis=0)
    design = pd.DataFrame(design)

    treatment_design_values = treatment_design.values
    design[treatment_design.columns] = np.tile(treatment_design_values, (len(plot_design), 1))

    return design


def generate_graeco_latin_square_design(variables, levels):
    check_input(variables, [], levels=levels)

    n = len(variables)
    matrix = np.zeros((n, n))
    for i, variable in enumerate(variables):
        matrix[i] = np.roll(np.arange(levels), i)
    design = np.zeros((n ** 2, n))
    for i in range(n):
        design[i * n: (i + 1) * n] = np.roll(matrix, i, axis=1)
    return pd.DataFrame(design, columns=variables)


def generate_youden_square_design(variables, levels):
    # Input validation
    check_input(variables, [], levels=levels)

    n = len(variables)
    matrix = np.zeros((n, n))
    for i, variable in enumerate(variables):
        matrix[i] = np.roll(np.arange(levels), i)
    design = np.zeros((n ** 2, n))
    for i in range(n):
        design[i * n: (i + 1) * n] = np.roll(matrix, i, axis=0)
    return pd.DataFrame(design, columns=variables)


def generate_completely_randomized_design(variables, levels, n_points):
    # Input validation
    check_input(variables, levels, n_points=n_points)

    matrix = np.zeros((n_points, len(variables)))
    for i, variable in enumerate(variables):
        matrix[:, i] = np.random.choice(levels[i], size=n_points)
    return pd.DataFrame(matrix, columns=variables)


def generate_randomized_block_design(variables, levels, n_blocks):
    # Input validation
    check_input(variables, levels, n_blocks=n_blocks)

    block_design = generate_completely_randomized_design(['Block'], range(n_blocks), n_blocks)
    matrix = np.zeros((n_blocks * len(variables), len(variables) + 1))
    for i, block in enumerate(block_design['Block']):
        start_row = i * len(variables)
        end_row = (i + 1) * len(variables)
        matrix[start_row:end_row, :-1] = generate_completely_randomized_design(variables, levels, len(variables))
        matrix[start_row:end_row, -1] = block
    return pd.DataFrame(matrix, columns=variables + ['Block'])


def generate_factorial_design_with_nested_factors(variables, levels, nested_variables, nested_levels):
    # Input validation
    check_input(variables, levels)
    check_input(nested_variables, nested_levels)

    nested_design = generate_completely_randomized_design(nested_variables, nested_levels, len(variables))
    design = generate_completely_randomized_design(variables, levels, len(variables))
    for nested_variable in nested_variables:
        design[nested_variable] = np.repeat(nested_design[nested_variable].values, len(variables) // len(nested_levels))
    return design


def generate_split_plot_fractional_factorial_design(variables, plot_factors, treatment_factors, factors):
    # Input validation
    check_input(variables, [], plot_factors=plot_factors, treatment_factors=treatment_factors, factors=factors)

    plot_design = generate_plackett_burman_design(variables[:plot_factors], plot_factors)
    treatment_design = generate_plackett_burman_design(variables[plot_factors:], treatment_factors)
    design = pd.DataFrame(np.repeat(plot_design.values, len(treatment_design), axis=0))
    for column in treatment_design.columns:
        design[column] = np.tile(treatment_design[column].values, len(plot_design))
    return design.iloc[:, :factors]


def generate_starting_design(n, k):
    D = np.random.uniform(-1, 1, size=(n, k))
    while np.linalg.matrix_rank(D) < min(n, k):  # Ensure non-singularity
        D = np.random.uniform(-1, 1, size=(n, k))
    return D



def construct_split_plot_design(num_runs, a_priori_model, num_factors, factor_types, constraints, num_starting_designs, num_rows, num_columns, var_ratios):
    # Generate starting design
    starting_design = generate_starting_design(num_rows, num_columns, factor_types, constraints)
    best_design = starting_design.copy()
    
    # Calculate initial D-optimality criterion value
    best_d_optimality = calculate_d_optimality(starting_design, a_priori_model)
    
    # Iterate for improvements
    for _ in range(num_starting_designs):
        design = starting_design.copy()
        d_optimality = best_d_optimality
        
        # Iterate over each element in the design matrix
        for i in range(num_rows):
            for j in range(num_columns):
                # Evaluate changes for factors applied to the rows
                if factor_types[i] == "continuous":
                    min_value, max_value = constraints[i]
                    for value in np.linspace(min_value, max_value, num=100):  # Adjust num for desired resolution
                        design[i, j] = value
                        d_optimality_temp = calculate_d_optimality(design, a_priori_model)
                        if d_optimality_temp > d_optimality:
                            d_optimality = d_optimality_temp
                            best_design = design.copy()
                elif factor_types[i] == "categorical":
                    possible_levels = constraints[i]
                    for level in possible_levels:
                        design[i, j] = level
                        d_optimality_temp = calculate_d_optimality(design, a_priori_model)
                        if d_optimality_temp > d_optimality:
                            d_optimality = d_optimality_temp
                            best_design = design.copy()
                
                # Evaluate changes for factors applied to the columns
                if factor_types[j + num_rows] == "continuous":
                    min_value, max_value = constraints[j + num_rows]
                    for value in np.linspace(min_value, max_value, num=100):  # Adjust num for desired resolution
                        design[i, j] = value
                        d_optimality_temp = calculate_d_optimality(design, a_priori_model)
                        if d_optimality_temp > d_optimality:
                            d_optimality = d_optimality_temp
                            best_design = design.copy()
                elif factor_types[j + num_rows] == "categorical":
                    possible_levels = constraints[j + num_rows]
                    for level in possible_levels:
                        design[i, j] = level
                        d_optimality_temp = calculate_d_optimality(design, a_priori_model)
                        if d_optimality_temp > d_optimality:
                            d_optimality = d_optimality_temp
                            best_design = design.copy()
        
        # Update the starting design for the next iteration
        starting_design = best_design.copy()
        best_d_optimality = d_optimality
    
    return best_design

def generate_starting_design(num_rows, num_columns, factor_types, constraints):
    design = np.zeros((num_rows, num_columns))
    
    for i in range(num_rows):
        if factor_types[i] == "continuous":
            min_value, max_value = constraints[i]
            design[i, :] = np.random.uniform(min_value, max_value, size=num_columns)
        elif factor_types[i] == "categorical":
            possible_levels = constraints[i]
            design[i, :] = np.random.choice(possible_levels, size=num_columns)
    
    for j in range(num_columns):
        if factor_types[j + num_rows] == "continuous":
            min_value, max_value = constraints[j + num_rows]
            design[:, j] = np.random.uniform(min_value, max_value, size=num_rows)
        elif factor_types[j + num_rows] == "categorical":
            possible_levels = constraints[j + num_rows]
            design[:, j] = np.random.choice(possible_levels, size=num_rows)
    
    return design


def coordinate_exchange_algorithm(n, k, beta, max_iter=1000, num_starts=1000):
    best_design = None
    best_optimality = -np.inf
    
    for _ in range(num_starts):
        D = generate_starting_design(n, k)
        
        for _ in range(max_iter):
            old_D = D.copy()
            
            for i in range(n):
                for j in range(k):
                    # Try changing to -1 and +1
                    for new_value in [-1, 1]:
                        D_trial = D.copy()
                        D_trial[i, j] = new_value
                        trial_optimality = calculate_d_optimality(D_trial, beta)
                        
                        # If this improves optimality, keep the change
                        if trial_optimality > best_optimality:
                            best_optimality = trial_optimality
                            D = D_trial
            
            # If no changes made in an iteration, stop
            if np.all(D == old_D):
                break
        
        # If this design is better than previous designs, keep it
        if calculate_d_optimality(D, beta) > best_optimality:
            best_optimality = calculate_d_optimality(D, beta)
            best_design = D
    
    return best_design

def coordinate_exchange_algorithm_augmentation(n2, k, beta, X1, max_iter=1000, num_starts=1000):
    best_design = None
    best_optimality = -np.inf
    
    for _ in range(num_starts):
        X2 = generate_starting_design(n2, k)
        X = np.vstack([X1, X2])
        
        for _ in range(max_iter):
            old_X = X.copy()
            
            for i in range(n2):
                for j in range(k):
                    # Try changing to -1 and +1
                    for new_value in [-1, 1]:
                        X_trial = X.copy()
                        X_trial[i + len(X1), j] = new_value
                        trial_optimality = calculate_d_optimality(X_trial, beta)
                        
                        # If this improves optimality, keep the change
                        if trial_optimality > best_optimality:
                            best_optimality = trial_optimality
                            X = X_trial
            
            # If no changes made in an iteration, stop
            if np.all(X == old_X):
                break
        
        # If this design is better than previous designs, keep it
        if calculate_d_optimality(X, beta) > best_optimality:
            best_optimality = calculate_d_optimality(X, beta)
            best_design = X
    
    return best_design


"""
Extensions section: tests, optimality measures, power calculations
"""


# Non-Parametric Tests
def nonparametric_kruskal_wallis(D, adjusted_alpha, one_sided):
    groups = D  # the data contains the different groups
    H, p_val = kruskal(*groups)
    return p_val < adjusted_alpha

def nonparametric_mann_whitney(D, adjusted_alpha, one_sided):
    sample1, sample2 = D  # unpack the data
    U, p_val = mannwhitneyu(sample1, sample2, alternative='two-sided' if not one_sided else 'less')
    return p_val < adjusted_alpha

def nonparametric_wilcoxon(D, adjusted_alpha, one_sided):
    sample1, sample2 = D  # unpack the data
    T, p_val = wilcoxon(sample1, sample2, alternative='two-sided' if not one_sided else 'less')
    return p_val < adjusted_alpha

# Parametric Tests
def anova(D, adjusted_alpha, one_sided):
    groups = D  # the data contains the different groups
    F, p_val = f_oneway(*groups)
    if one_sided:
        p_val /= 2
    return p_val < adjusted_alpha

def chi_squared_test(D, adjusted_alpha, one_sided):
    contingency_table = D  # the data is a contingency table
    chi2_stat, p_val, dof, ex = chi2_contingency(contingency_table)
    return p_val < adjusted_alpha

def one_sample_t_test(D, adjusted_alpha, one_sided):
    sample, popmean = D  # unpack the data
    t_stat, p_val = ttest_1samp(sample, popmean)
    if one_sided:
        p_val /= 2
    return p_val < adjusted_alpha

def two_sample_t_test(D, adjusted_alpha, one_sided):
    sample1, sample2 = D  # unpack the data
    t_stat, p_val = ttest_ind(sample1, sample2)
    if one_sided:
        p_val /= 2
    return p_val < adjusted_alpha

def paired_t_test(D, adjusted_alpha, one_sided):
    sample1, sample2 = D  # unpack the data
    t_stat, p_val = ttest_rel(sample1, sample2)
    if one_sided:
        p_val /= 2
    return p_val < adjusted_alpha


def hypothesis_test(D, alpha, one_sided):
    # Perform a one-sample t-test against the null hypothesis
    
    # Example: Perform a one-sample t-test against the null hypothesis of mean = 0
    t_statistic, p_value = stats.ttest_1samp(D, 0)
    
    if one_sided:
        # For one-sided test, check if p-value is less than the adjusted alpha
        return p_value < alpha
    else:
        # For two-sided test, check if p-value is less than half of the adjusted alpha
        return p_value < alpha / 2


def calculate_information_matrix(D, beta):
    return D.T @ D

def calculate_d_optimality(D, beta):
    return np.linalg.det(calculate_information_matrix(D, beta))

def d_optimality(X):
    XTX_inv = inv(X.T @ X)
    return np.linalg.det(XTX_inv)

def e_optimality(X):
    info_matrix = X.T @ X
    eigenvalues = eigvals(info_matrix)
    return np.max(eigenvalues)

def s_optimality(X):
    mutual_orthogonality = np.linalg.norm(X.T @ X - np.eye(X.shape[1]))
    det_info_matrix = np.linalg.det(X.T @ X)
    return mutual_orthogonality * det_info_matrix

def g_optimality(X):
    hat_matrix = X @ inv(X.T @ X) @ X.T
    return np.max(np.diag(hat_matrix))


def calculate_power(n, k, beta, alpha=0.05, num_simulations=1000, multiple_comparison=False, one_sided=False, correction_method='bonferroni'):
    power = 0
    num_tests = 1  # Number of hypothesis tests being performed
    
    if multiple_comparison:
        num_tests = k  # Update the number of tests based on the number of variables
    
    if correction_method == 'bonferroni':
        adjusted_alpha = alpha / num_tests  # Bonferroni correction
    elif correction_method == 'sidak':
        adjusted_alpha = 1 - (1 - alpha) ** (1 / num_tests)  # Šidák correction
    
    for _ in range(num_simulations):
        D = coordinate_exchange_algorithm(n, k, beta)
        
        # Perform hypothesis test and check if null hypothesis is rejected
        # based on your specific analysis and research question.
        if hypothesis_test(D, adjusted_alpha, one_sided):
            power += 1
    
    power = power / num_simulations * 100  # Calculate power as a percentage
    
    return power


"""
Plotting section, hopefully to be expanded
"""


"""

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sigma2_epsilon = 1.0

generate_fds_plot(X, sigma2_epsilon, num_points=10000)

"""

def f(x):
    """Function to expand vector of factor settings"""
    x1i, x2i, x3i = x
    return np.array([
        1, x1i, x2i, x3i, x1i*x2i, x1i*x3i, x2i*x3i, x1i*x2i*x3i,
        x1i**2, x2i**2, x3i**2
    ])

def calculate_relative_variance(X, sigma2_epsilon):
    """Calculate relative variance of prediction for each point in X"""
    X_transpose = np.transpose(X)
    XTX_inv = np.linalg.inv(np.dot(X_transpose, X))
    f_values = np.apply_along_axis(f, 1, X)
    variances = np.diag(np.dot(np.dot(f_values, XTX_inv), f_values.T))
    return variances / sigma2_epsilon

def generate_fds_plot(X, sigma2_epsilon, num_points=10000):
    """Generate FDS plot for a given design"""
    sample_points = np.random.uniform(size=(num_points, X.shape[1]))
    relative_variances = calculate_relative_variance(sample_points, sigma2_epsilon)
    sorted_indices = np.argsort(relative_variances)
    sorted_variances = relative_variances[sorted_indices]
    normalized_indices = np.linspace(0, 1, num_points)
    
    plt.plot(normalized_indices, sorted_variances)
    plt.xlabel('Fraction of Design Space')
    plt.ylabel('Relative Variance of Prediction')
    plt.title('FDS Plot')
    plt.show()





