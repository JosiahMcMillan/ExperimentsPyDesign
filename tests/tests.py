import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product
from .designs import (
    check_input,
    generate_taguchi_design,
    generate_rsm_design,
    generate_plackett_burman_design,
    generate_ccd_design,
    generate_box_behnken_design,
    generate_augmented_design,
    generate_split_plot_design,
    generate_graeco_latin_square_design,
    generate_youden_square_design,
    generate_completely_randomized_design,
    generate_randomized_block_design,
    generate_factorial_design_with_nested_factors,
    generate_split_plot_fractional_factorial_design,
    generate_starting_design,
    construct_split_plot_design,
    # TODO: the following algos
    # coordinate_exchange_algorithm,
    # coordinate_exchange_algorithm_augmentation,
)

class TestYourCode(unittest.TestCase):
    def test_check_input(self):
        # Test valid input
        variables = ['Variable1', 'Variable2', 'Variable3']
        levels = [[-1, 1], [-1, 1], [-1, 1]]
        self.assertIsNone(check_input(variables, levels))

        # Test invalid input: variables and levels have different lengths
        variables = ['Variable1', 'Variable2', 'Variable3']
        levels = [[-1, 1], [-1, 1]]
        self.assertRaises(ValueError, check_input, variables, levels)

        # Test invalid input: n_points is not an integer
        variables = ['Variable1', 'Variable2', 'Variable3']
        levels = [[-1, 1], [-1, 1], [-1, 1]]
        n_points = 10.5
        self.assertRaises(TypeError, check_input, variables, levels, n_points)

    def test_generate_taguchi_design(self):
        # Test generating a Taguchi design
        variables = ['Variable1', 'Variable2']
        levels = [[-1, 1], [-1, 1]]
        design = generate_taguchi_design(variables, levels)

        # Check the design dimensions
        self.assertEqual(design.shape, (4, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = list(product(*levels))
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_rsm_design(self):
        # Test generating an RSM design
        variables = ['Variable1', 'Variable2', 'Variable3']
        ranges = [[-1, 1], [-1, 1], [-1, 1]]
        design = generate_rsm_design(variables, ranges)

        # Check the design dimensions
        self.assertEqual(design.shape, (27, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = list(product(*ranges))
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_plackett_burman_design(self):
        # Test generating a Plackett-Burman design
        variables = ['Variable1', 'Variable2', 'Variable3']
        factors = 2
        design = generate_plackett_burman_design(variables, factors)

        # Check the design dimensions
        self.assertEqual(design.shape, (4, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = [[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
        actual_combinations = list(map(list, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_ccd_design(self):
        # Test generating a CCD design
        variables = ['Variable1', 'Variable2', 'Variable3']
        ranges = [[-1, 1], [-1, 1], [-1, 1]]
        design = generate_ccd_design(variables, ranges)

        # Check the design dimensions
        self.assertEqual(design.shape, (9, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [0, 0, 0]]
        actual_combinations = list(map(list, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_box_behnken_design(self):
        # Test generating a Box-Behnken design
        variables = ['Variable1', 'Variable2', 'Variable3']
        ranges = [[-1, 1], [-1, 1], [-1, 1]]
        design = generate_box_behnken_design(variables, ranges)

        # Check the design dimensions
        self.assertEqual(design.shape, (13, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]
        ]
        actual_combinations = list(map(list, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_augmented_design(self):
        # Test generating an augmented design
        variables = ['Variable1', 'Variable2']
        factors = 2
        interactions = 2
        design = generate_augmented_design(variables, factors, interactions)

        # Check the design dimensions
        self.assertEqual(design.shape, (6, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the levels are -1 or 1
        levels = [-1, 1]
        for column in design.columns:
            unique_levels = design[column].unique()
            self.assertListEqual(list(unique_levels), levels)

    def test_generate_split_plot_design(self):
        # Test generating a split-plot design
        variables = ['Plot', 'Treatment']
        plot_factors = 2
        treatment_factors = 3
        design = generate_split_plot_design(variables, plot_factors, treatment_factors)

        # Check the design dimensions
        self.assertEqual(design.shape, (12, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_plot_combinations = list(product([-1, 1], repeat=plot_factors))
        expected_treatment_combinations = list(product([-1, 1], repeat=treatment_factors))
        expected_combinations = list(product(expected_plot_combinations, expected_treatment_combinations))
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_graeco_latin_square_design(self):
        # Test generating a Graeco-Latin square design
        variables = ['Variable1', 'Variable2']
        levels = [3, 3]
        design = generate_graeco_latin_square_design(variables, levels)

        # Check the design dimensions
        self.assertEqual(design.shape, (9, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = [
            (0, 0), (0, 1), (0, 2),
            (1, 1), (1, 2), (1, 0),
            (2, 2), (2, 0), (2, 1)
        ]
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_youden_square_design(self):
        # Test generating a Youden square design
        variables = ['Variable1', 'Variable2']
        levels = [3, 3]
        design = generate_youden_square_design(variables, levels)

        # Check the design dimensions
        self.assertEqual(design.shape, (9, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_combinations = [
            (0, 0), (1, 1), (2, 2),
            (1, 0), (2, 1), (0, 2),
            (2, 0), (0, 1), (1, 2)
        ]
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_completely_randomized_design(self):
        # Test generating a completely randomized design
        variables = ['Variable1', 'Variable2', 'Variable3']
        levels = [[-1, 0, 1], [-1, 1], [-1, 1]]
        n_points = 10
        design = generate_completely_randomized_design(variables, levels, n_points)

        # Check the design dimensions
        self.assertEqual(design.shape, (n_points, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the levels are within the specified ranges
        for i, column in enumerate(design.columns):
            unique_levels = design[column].unique()
            self.assertListEqual(list(unique_levels), levels[i])

    def test_generate_randomized_block_design(self):
        # Test generating a randomized block design
        variables = ['Variable1', 'Variable2']
        levels = [[-1, 1], [-1, 1]]
        n_blocks = 3
        design = generate_randomized_block_design(variables, levels, n_blocks)

        # Check the design dimensions
        self.assertEqual(design.shape, (6, 3))

        # Check the column names
        self.assertListEqual(list(design.columns), variables + ['Block'])

        # Check the levels are within the specified ranges
        for i, column in enumerate(design.columns[:-1]):
            unique_levels = design[column].unique()
            self.assertListEqual(list(unique_levels), levels[i])

        # Check the number of unique blocks
        unique_blocks = design['Block'].unique()
        self.assertEqual(len(unique_blocks), n_blocks)

    def test_generate_factorial_design_with_nested_factors(self):
        # Test generating a factorial design with nested factors
        variables = ['Variable1', 'Variable2']
        levels = [[-1, 1], [-1, 1]]
        nested_variables = ['Nested1', 'Nested2']
        nested_levels = [[0, 1], [0, 1]]
        design = generate_factorial_design_with_nested_factors(variables, levels, nested_variables, nested_levels)

        # Check the design dimensions
        self.assertEqual(design.shape, (4, 4))

        # Check the column names
        self.assertListEqual(list(design.columns), variables + nested_variables)

        # Check the levels are within the specified ranges
        for i, column in enumerate(design.columns[:-2]):
            unique_levels = design[column].unique()
            self.assertListEqual(list(unique_levels), levels[i])

        # Check the levels of the nested factors
        for i, column in enumerate(design.columns[-2:]):
            unique_levels = design[column].unique()
            self.assertListEqual(list(unique_levels), nested_levels[i])

    def test_generate_split_plot_fractional_factorial_design(self):
        # Test generating a split-plot fractional factorial design
        variables = ['Plot', 'Treatment']
        plot_factors = 2
        treatment_factors = 2
        factors = 2
        design = generate_split_plot_fractional_factorial_design(variables, plot_factors, treatment_factors, factors)

        # Check the design dimensions
        self.assertEqual(design.shape, (4, 2))

        # Check the column names
        self.assertListEqual(list(design.columns), variables)

        # Check the unique combinations of levels
        expected_plot_combinations = list(product([-1, 1], repeat=plot_factors))
        expected_treatment_combinations = list(product([-1, 1], repeat=treatment_factors))
        expected_combinations = list(product(expected_plot_combinations, expected_treatment_combinations))[:factors]
        actual_combinations = list(map(tuple, design.values))
        self.assertListEqual(actual_combinations, expected_combinations)

    def test_generate_starting_design(self):
        # Test generating a starting design
        n = 5
        k = 3
        design = generate_starting_design(n, k)

        # Check the design dimensions
        self.assertEqual(design.shape, (n, k))

        # Check the design matrix rank
        rank = np.linalg.matrix_rank(design)
        self.assertEqual(rank, min(n, k))

    def test_construct_split_plot_design(self):
        # Test constructing a split-plot design
        num_runs = 6
        a_priori_model = 2
        num_factors = 2
        factor_types = ['continuous', 'categorical', 'continuous']
        constraints = [[-1, 1], [0, 1, 2], [-1, 1]]
        num_starting_designs = 100
        num_rows = 3
        num_columns = 2
        var_ratios = [1.0, 1.0, 1.0]
        design = construct_split_plot_design(
            num_runs, a_priori_model, num_factors, factor_types, constraints,
            num_starting_designs, num_rows, num_columns, var_ratios
        )

        # Check the design dimensions
        self.assertEqual(design.shape, (num_rows + num_runs, num_columns))

        # Check the design matrix rank
        rank = np.linalg.matrix_rank(design)
        self.assertEqual(rank, min(num_rows + num_runs, num_columns))


if __name__ == '__main__':
    unittest.main()
