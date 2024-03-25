from config.schemas import raw_schema
import polars as pl
from pathlib import Path
from logging import Logger
from config.utils import tce_logger


class DataCleaning:
    """
    Cleans TCE-RS data and filters necessary columns.

    Attributes
    ----------
    data_dir: Path, defaults to data/tce_licitations.csv
        CSV Data directory.
    logger: Logger, defaults to tce_logger
        Logger.
    """

    def __init__(
        self, data_dir: Path = "data/tce_licitations.csv", logger: Logger = tce_logger
    ):
        self.data_dir = data_dir
        self.logger = logger

    def load(self) -> pl.DataFrame:
        """
        Loads data.

        Returns
        -------
        df: pl.DataFrame
            TCE data.
        """

        df_raw = pl.read_csv(self.data_dir, schema=raw_schema).lazy()

        cols_to_keep = [
            "CD_ORGAO",
            "NM_ORGAO",
            "ANO_LICITACAO",
            "DS_OBJETO",
            "VL_LICITACAO",
            "DT_HOMOLOGACAO",
            "VL_HOMOLOGADO",
        ]

        df_raw = df_raw.select(cols_to_keep).unique()

        self.logger.info("Data loaded!")
        return df_raw

    def asserts_data_types(self, df_raw: pl.DataFrame) -> pl.DataFrame:
        """
        Asserting the correct data types.

        Parameters
        ----------
        df_raw: pl.DataFrame
            TCE data with filtered columns.

        Returns
        ----------
        df_with_dtypes: pl.DataFrame
            pl.DataFrame with correct data types.
        """

        # Replaces a few values
        df_raw = df_raw.filter(~pl.col("ANO_LICITACAO").is_in(["PRD", "PDE"]))
        df_raw = df_raw.with_columns(
            pl.col("ANO_LICITACAO").replace({"2023.0": "2023", "2024.0": "2024"})
        )
        df_raw = df_raw.filter(pl.col("DS_OBJETO") != "REGISTRO DE PREÇOS DE INSUMOS ")

        # Asserting the correct data types
        df_with_dtypes = df_raw.with_columns(pl.col("ANO_LICITACAO").cast(pl.Int64))
        df_with_dtypes = df_with_dtypes.with_columns(pl.col("CD_ORGAO").cast(pl.Int64))
        df_with_dtypes = df_with_dtypes.with_columns(
            pl.col("ANO_LICITACAO").cast(pl.Int64)
        )
        df_with_dtypes = df_with_dtypes.with_columns(
            pl.col("VL_HOMOLOGADO").cast(pl.Float64)
        )
        df_with_dtypes = df_with_dtypes.with_columns(
            pl.col("DT_HOMOLOGACAO")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias("DT_HOMOLOGACAO")
        )

        self.logger.info("Data types asserted!")
        return df_with_dtypes

    def clean_nan(self, df_with_dtypes: pl.DataFrame) -> pl.DataFrame:
        """
        Selects only necessary columns and replaces null values.

        Parameters
        ----------
        df_with_dtypes: pl.DataFrame
            pl.DataFrame with correct data types.

        Returns
        -------
        df_final: pl.DataFrame
            Final TCE data.
        """

        df_final = df_with_dtypes.with_columns(
            pl.col("VL_HOMOLOGADO").fill_null(pl.col("VL_LICITACAO"))
        )

        self.logger.info("Null values cleaned!")
        return df_final

    def run(self) -> pl.DataFrame:
        """
        Executes the full process

        Returns
        ----------
        df_final: pl.DataFrame
            Final TCE data.
        """

        df_raw = self.load()
        df_with_dtypes = self.asserts_data_types(df_raw)
        df_final = self.clean_nan(df_with_dtypes)

        self.logger.info("Full data cleaned!")

        return df_final
