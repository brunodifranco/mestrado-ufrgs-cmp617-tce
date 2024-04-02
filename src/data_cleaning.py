from utils.schemas import raw_dtypes
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path
from logging import Logger
from utils.utils import logger


class DataCleaning:
    """
    Cleans TCE-RS data.

    Attributes
    ----------
    data_dir: Path, defaults to data/tce_licitations.csv
        CSV Data directory.
    logger: Logger, defaults to logger
        Logger.
    """

    def __init__(
        self, data_dir: Path = "data/tce_licitations.csv", logger: Logger = logger
    ):
        self.data_dir = data_dir
        self.logger = logger

    def load(self) -> DataFrame:
        """
        Loads data.

        Returns
        -------
        df: DataFrame
            TCE data.
        """

        df = pd.read_csv(self.data_dir, dtype=raw_dtypes)
        self.logger.info("Data loaded!")

        return df

    def clean_nan(self, df: DataFrame) -> DataFrame:
        """
        Selects only necessary columns and replaces null values.

        Parameters
        ----------
        df: DataFrame
            TCE data.

        Returns
        -------
        df_cleaned_nan: DataFrame
            TCE data with cleaned null values.
        """

        cols_to_keep = [
            "CD_ORGAO",
            "NM_ORGAO",
            "ANO_LICITACAO",
            "DS_OBJETO",
            "VL_LICITACAO",
            "DT_HOMOLOGACAO",
            "VL_HOMOLOGADO",
        ]
        df_cleaned_nan = df[cols_to_keep].copy()

        df_cleaned_nan.loc[:, "VL_HOMOLOGADO"] = df_cleaned_nan["VL_HOMOLOGADO"].fillna(
            df_cleaned_nan["VL_LICITACAO"]
        )

        df_cleaned_nan = df_cleaned_nan.dropna(subset=["ANO_LICITACAO"])

        self.logger.info("Null values cleaned!")
        return df_cleaned_nan

    def asserts_data_types(self, df_cleaned_nan) -> DataFrame:
        """
        Asserting the correct data types.

        Parameters
        ----------
        df_cleaned_nan: DataFrame
            TCE data with cleaned null values.

        Returns
        ----------
        df_final: DataFrame
            Cleaned DataFrame.
        """

        # Replaces a few values        
        df_final = df_cleaned_nan[~df_cleaned_nan["ANO_LICITACAO"].isin(["PRD", "PDE"])]
        df_final.loc[:, "ANO_LICITACAO"] = df_final["ANO_LICITACAO"].replace(
            {"2023.0": "2023", "2024.0": "2024"}
        )
        df_final = df_final[
            ~df_final["VL_HOMOLOGADO"].isin(["###############", "#################"])
        ]

        # Assert data types
        df_final["CD_ORGAO"] = df_final["CD_ORGAO"].astype(int)
        df_final["ANO_LICITACAO"] = df_final["ANO_LICITACAO"].astype(int)
        df_final["VL_LICITACAO"] = df_final["VL_LICITACAO"].astype(float)
        df_final["DT_HOMOLOGACAO"] = pd.to_datetime(df_final["DT_HOMOLOGACAO"])
        df_final["VL_HOMOLOGADO"] = df_final["VL_HOMOLOGADO"].astype(float)

        self.logger.info("Data types asserted!")
        return df_final

    def run(self) -> DataFrame:
        """
        Executes the full process

        Returns
        ----------
        df_final: DataFrame
            Cleaned DataFrame.
        """

        df = self.load()
        df_cleaned_nan = self.clean_nan(df)
        df_final = self.asserts_data_types(df_cleaned_nan)

        self.logger.info("Full data cleaned!")

        return df_final
