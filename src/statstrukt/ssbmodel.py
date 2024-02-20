import numpy as np
import pandas as pd


class ssbmodel:
    """Class for model estimation"""

    def __init__(
        self, pop_data: pd.DataFrame, sample_data: pd.DataFrame, id_nr: str
    ) -> None:
        self.pop_data = pop_data
        self.sample_data = sample_data

        # Check id variable - can be numeric or character
        self._check_variable(
            id_nr,
            self.pop_data,
            data_name="population",
            check_for_char=True,
            remove_missing=False,
        )
        self._check_variable(
            id_nr, self.sample_data, check_for_char=True, remove_missing=False
        )
        self.id_nr = id_nr

    def __call__(self):
        print(
            "strukturmodel instance with data including population size "
            + self.pop.shape[0]
            + " and sample size "
            + self.sample_data.shape[0]
        )

    def _check_variable(
        self,
        var_name,
        dataset,
        data_name="sample",
        check_for_char=False,
        remove_missing=False,
    ):
        """Check if the given variable name is in the dataset

        Args:
            var_name: str, the name of the variable to check.
            dataset: dict, the dataset where keys are variable names and values are the data.
            data_name: str,
            check_for_id: bool, False by default. If True, checks for an ID variable that can be numeric or character.

        Returns:
            None: The function will either pass silently or raise an error.

        Raises:
            ValueError: If the variable is not numeric (when not checking for ID) or if the variable doesn't exist in the dataset.
        """
        # Check if var_name is a key in the dataset
        if var_name not in dataset:
            raise ValueError(
                f"Variable '{var_name}' not found in the {data_name} dataset."
            )

        # If checking for an ID variable, just check for existence
        if check_for_char:
            return  # Variable exists, no further checks needed

        # If not checking for an ID, verify that the variable is numeric
        value = dataset[var_name]

        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(
                f"Variable '{var_name}' in the {data_name} dataset needs to be numeric but isn't."
            )

        # Check and remove observations with missing values
        if remove_missing:
            cleaned_dataframe = dataset.dropna(subset=var_name)
            missing_num = dataset.shape[0] - cleaned_dataframe.shape[0]
            if missing_num > 0:
                print(
                    f"There were {missing_num} missing values in variable {var_name}. These observations were reomved from the sample"
                )
            self.sample_data = cleaned_dataframe
        else:

            # Give error for missing values
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise ValueError(
                    f"Variable '{var_name}' contains missing values. Please fix and try again."
                )

    def _add_flag(self):
        """Add flag in population data to say if unit is in the sample or not."""
        self.flag_var = "_flag_sample"
        sample_ids = set(self.sample_data[self.id_nr])
        self.pop_data[self.flag_var] = self.pop_data[self.id_nr].apply(
            lambda x: 1 if x in sample_ids else 0
        )

    def _check_model_run(self):
        """Check to ensure that model has been run before proceeding with other functions"""
        if not hasattr(self, "strata_results"):
            raise RuntimeError("Model has not been run. Please run fit() first")
