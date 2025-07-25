import pandas as pd


class ssbmodel:
    """Class for model estimation."""

    def __init__(
        self,
        pop_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
    ) -> None:
        """Initialize general ssbmodel object.

        Args:
            pop_data: Population data with one row per unit (company)
            sample_data: Sample data containing variables on the interest variable(s).
            id_nr: Name of the variable to identify units in the population and sample
            verbose: Whether to show printed output or not. 1 is minimal output, 2 is more output.
        """
        self.pop_data = pop_data
        self.sample_data = sample_data

        # Check id variable in population and sample - can be numeric or character
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
        self._check_id_duplicates()

        self.verbose = verbose

    def __call__(self) -> None:
        """Print model object."""
        print(
            f"strukturmodel instance with data including population size {self.pop_data.shape[0]} and sample size {self.sample_data.shape[0]}"
        )

    def change_verbose(self, verbose: int) -> None:
        """Change the verbose print level."""
        self.verbose = verbose

    def _check_variable(
        self,
        var_name: str,
        dataset: pd.DataFrame,
        data_name: str = "sample",
        check_for_char: bool = False,
        remove_missing: bool = True,
    ) -> None:
        """Check if the given variable name is in the dataset.

        Args:
            var_name: The name of the variable to check.
            dataset: The dataset where keys are variable names and values are the data.
            data_name: Name of the dataset to use in error message
            check_for_char: False by default. If True, checks for an ID variable that can be numeric or character.
            remove_missing: Whether to remove missing matches. Default True.

        Raises:
            ValueError: If the variable is not numeric (when not checking for ID) or if the variable doesn't exist in the dataset.
        """
        # Check if var_name is a key in the dataset
        if var_name not in dataset:
            raise ValueError(
                f"Variable '{var_name}' not found in the {data_name} dataset."
            )

        value = dataset[var_name]

        # If not checking for an ID, verify that the variable is numeric
        if not check_for_char:
            if not pd.api.types.is_numeric_dtype(value):
                raise ValueError(
                    f"Variable '{var_name}' in the {data_name} dataset needs to be numeric but isn't."
                )
            if any(value < 0):
                raise ValueError(
                    f"There are negative values in the variable '{var_name}' in the {data_name} dataset. Consider a log transformation or another type of model."
                )

        # Check and remove observations with missing values
        if (remove_missing) & (data_name == "sample"):
            cleaned_dataframe = dataset.dropna(subset=var_name)
            missing_num = dataset.shape[0] - cleaned_dataframe.shape[0]
            if missing_num > 0:
                print(
                    f"There were {missing_num} missing values in variable {var_name}. These observations were removed from the sample."
                )
            self.sample_data = cleaned_dataframe
        else:
            # Give error for missing values
            if value.isna().any():
                raise ValueError(
                    f"Variable '{var_name}' contains missing values. Please fix and try again."
                )

    def _check_id_duplicates(self) -> None:
        """Check sample and population data for duplicate ids."""
        duplicate_pop = self.pop_data[self.id_nr].duplicated().any()
        duplicate_sample = self.sample_data[self.id_nr].duplicated().any()
        if duplicate_pop or duplicate_sample:
            raise ValueError(
                f"Duplicates found in {'pop_data' if duplicate_pop else ''}{' and ' if duplicate_pop and duplicate_sample else ''}{'sample_data' if duplicate_sample else ''} based on {self.id_nr}. Please fix before proceeding."
            )

    def _check_model_run(self) -> None:
        """Check to ensure that model has been run before proceeding with other functions."""
        if not hasattr(self, "strata_results"):
            raise RuntimeError("Model has not been run. Please run fit() first")

    def _convert_var(self, var_name: str, dataset: pd.DataFrame) -> None:
        if dataset[var_name].dtype == "Int64":
            dataset[var_name] = dataset[var_name].astype("int64")
        if dataset[var_name].dtype == "Float64":
            dataset[var_name] = dataset[var_name].astype("float64")
