import logging

import pandas as pd


class BaseModel:
    """Class for model estimation."""

    def __init__(
        self,
        pop_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        id_nr: str,
        verbose: int = 1,
        logger_level: str = "warning",
    ) -> None:
        """Initialize general base model object.

        Args:
            pop_data: Population data with one row per unit (company)
            sample_data: Sample data containing variables on the interest variable(s).
            id_nr: Name of the variable to identify units in the population and sample
            verbose: Whether to show printed output or not. 1 is minimal output, 2 is more output. This is for back compatibility only. Use logger_level instead.
            logger_level: Level of logging, Choose between "debug", "info", "warning","error", or "critical".
        """
        self.pop_data = pop_data
        self.sample_data = sample_data

        # Start logging
        self.logger = logging.getLogger("model")
        self.logger.setLevel(self.logging_dict[logger_level.lower()])

        # add in console handling
        if not self.logger.handlers:  # Avoid adding multiple handlers
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.logging_dict[logger_level])
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Change logger level from default if a different verbose is given on initialization
        if verbose != 1:
            self.change_verbose(verbose=verbose)

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
            f"strukturmodel instance with data including population size {self.pop_data.shape[0]} and sample size {self.sample_data.shape[0]}."
        )

    def change_verbose(self, verbose: int) -> None:
        """Change the verbose print level."""
        self.verbose = verbose
        if verbose <= 1:
            self.change_logging_level(logger_level="warning")
        if verbose > 1:
            self.change_logging_level(logger_level="info")

    def change_logging_level(self, logger_level: str) -> None:
        """Change the logging print level.

        Args:
            logger_level: Detail level for information output. Choose between 'debug','info','warning','error' and 'critical'.
        """
        self.logger.setLevel(self.logging_dict[logger_level])

    @property
    def logging_dict(self) -> dict[str, int]:
        """Returns a dictionary mapping standard logging level names to their corresponding numeric values.

        Returns:
            dict[str, int]: A dictionary of logging level names and their numeric values.
        """
        logging_dict = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        }
        return logging_dict

    @property
    def get_logging_level(self) -> str:
        """Retrieves the name of the current logging level set in the logger.

        Returns:
            str: The name of the current logging level, or a message if none is set.
        """
        level_name: str | None = next(
            (k for k, v in self.logging_dict.items() if v == self.logger.level), None
        )
        if level_name is None:
            level_name = "No logging level set"
        return level_name

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
            mes_var: str = (
                f"Variable '{var_name}' not found in the {data_name} dataset."
            )
            self.logger.error(msg=mes_var)
            raise ValueError(mes_var)

        value = dataset[var_name]

        # If not checking for an ID, verify that the variable is numeric
        if not check_for_char:
            if not pd.api.types.is_numeric_dtype(value):
                mes_char: str = (
                    f"Variable '{var_name}' in the {data_name} dataset needs to be numeric but isn't."
                )
                self.logger.error(msg=mes_char)
                raise ValueError(mes_char)
            if any(value < 0):
                mes_neg: str = (
                    f"There are negative values in the variable '{var_name}' in the {data_name} dataset. Consider a log transformation or another type of model."
                )
                self.logger.error(msg=mes_neg)
                raise ValueError(mes_neg)

        # Check and remove observations with missing values
        if (remove_missing) & (data_name == "sample"):
            cleaned_dataframe = dataset.dropna(subset=var_name)
            missing_num = dataset.shape[0] - cleaned_dataframe.shape[0]
            if missing_num > 0:
                mes_miss = f"There were {missing_num} missing values in variable {var_name}. These observations were removed from the sample."
                self.logger.warning(mes_miss)
            self.sample_data = cleaned_dataframe
        else:
            # Give error for missing values
            if value.isna().any():
                mes_miss2 = f"Variable '{var_name}' contains missing values. Please fix and try again."
                self.logger.error(mes_miss2)
                raise ValueError(mes_miss2)

    def _check_id_duplicates(self) -> None:
        """Check sample and population data for duplicate ids."""
        duplicate_pop = self.pop_data[self.id_nr].duplicated().any()
        duplicate_sample = self.sample_data[self.id_nr].duplicated().any()
        if duplicate_pop or duplicate_sample:
            mes_dup: str = (
                f"Duplicates found in {'pop_data' if duplicate_pop else ''}{' and ' if duplicate_pop and duplicate_sample else ''}{'sample_data' if duplicate_sample else ''} based on {self.id_nr}. Please fix before proceeding."
            )
            self.logger.error(msg=mes_dup)
            raise ValueError(mes_dup)

    def _check_model_run(self) -> None:
        """Check to ensure that model has been run before proceeding with other functions."""
        if not hasattr(self, "strata_results"):
            mes_run = "Model has not been run. Please run fit() first"
            self.logger.error(msg=mes_run)
            raise RuntimeError(mes_run)

    @staticmethod
    def _convert_var(var_name: str, dataset: pd.DataFrame) -> pd.Series:
        """Convert variable to int64 or float64 to be able to run model."""
        if dataset[var_name].dtype == "Int64":
            dataset[var_name] = dataset[var_name].astype("int64")
        if dataset[var_name].dtype == "Float64":
            dataset[var_name] = dataset[var_name].astype("float64")
        return dataset[var_name]
