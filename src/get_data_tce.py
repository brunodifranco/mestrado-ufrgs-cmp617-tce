import os
import zipfile
import shutil
import pandas as pd
from pandas.core.frame import DataFrame
from pathlib import Path
from logging import Logger
from config.utils import logger


class DataTCE:
    """
    Gets TCE-RS data.

    Attributes
    ----------
    data_dir: Path
        Data directory.
    logger: Logger
        Logger.
    """

    def __init__(self, data_dir: Path, logger: Logger):
        self.data_dir = data_dir
        self.logger = logger

    def downloads_data(self):
        """Downloads TCE data"""

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            os.chmod(self.data_dir, 0o755)

        years = list(range(2016, 2024))
        for year in years:
            url = f"https://dados.tce.rs.gov.br/dados/licitacon/licitacao/ano/{year}.csv.zip"
            output_file = os.path.join(self.data_dir, f"{year}.csv.zip")
            os.system(f"curl {url} --output {output_file}")

    def extract_files(self):
        """Unzip files"""
        arquivos_zip = [f for f in os.listdir(self.data_dir) if f.endswith(".zip")]

        for arquivo_zip in arquivos_zip:
            nome_arquivo = os.path.splitext(arquivo_zip)[0]
            pasta_destino = os.path.join(self.data_dir, nome_arquivo)

            if not os.path.exists(pasta_destino):
                os.makedirs(pasta_destino)

            caminho_arquivo_zip = os.path.join(self.data_dir, arquivo_zip)

            with zipfile.ZipFile(caminho_arquivo_zip, "r") as zip_ref:
                zip_ref.extractall(pasta_destino)

            self.logger.info(f"{arquivo_zip} extracted to {pasta_destino}.")

        self.logger.info("All files have been extracted.")

    def rename_csv_files(self):
        """Renames csv files"""

        def rename_licitation_csv(pasta: Path) -> Path:
            """
            Renames licitation csvs.

            Parameters
            ----------
            folder: Path
                Data folder.
            """

            caminho_licitacao_csv = os.path.join(pasta, "licitacao.csv")
            if os.path.exists(caminho_licitacao_csv):
                nome_pasta = os.path.basename(pasta)
                novo_nome_licitacao = f"licitacao_{nome_pasta}.csv"
                novo_caminho_licitacao = os.path.join(pasta, novo_nome_licitacao)
                os.rename(caminho_licitacao_csv, novo_caminho_licitacao)
                return novo_caminho_licitacao

        for pasta in os.listdir(self.data_dir):
            caminho_pasta = os.path.join(self.data_dir, pasta)
            if os.path.isdir(caminho_pasta):
                novo_caminho_licitacao = rename_licitation_csv(caminho_pasta)
                if novo_caminho_licitacao:
                    novo_caminho_licitacao_destino = os.path.join(
                        self.data_dir, os.path.basename(novo_caminho_licitacao)
                    )
                    shutil.move(novo_caminho_licitacao, novo_caminho_licitacao_destino)
                    self.logger.info(
                        f"File {os.path.basename(novo_caminho_licitacao)} moved to {self.data_dir}"
                    )

    def delete_non_licitation_files(self):
        """Deletes non licitation files"""

        def delete_files_not_licitation(pasta):
            """Deletes non licitation folders"""

            for arquivo in os.listdir(pasta):
                if arquivo != "licitacao_ano.csv":
                    caminho_arquivo = os.path.join(pasta, arquivo)
                    if os.path.isfile(caminho_arquivo):
                        os.remove(caminho_arquivo)
                        self.logger.info(f"File {caminho_arquivo} removed.")

        for pasta in os.listdir(self.data_dir):
            caminho_pasta = os.path.join(self.data_dir, pasta)
            if os.path.isdir(caminho_pasta):
                delete_files_not_licitation(caminho_pasta)

    def cleanup(self):
        """Cleans directory"""

        for item in os.listdir(self.data_dir):
            caminho_item = os.path.join(self.data_dir, item)

            if os.path.isfile(caminho_item):

                if not item.endswith(".csv"):
                    os.remove(caminho_item)
                    self.logger.info(f"File {item} removed.")

            elif os.path.isdir(caminho_item):

                if not item.endswith(".zip"):

                    for arquivo in os.listdir(caminho_item):
                        caminho_arquivo = os.path.join(caminho_item, arquivo)
                        os.remove(caminho_arquivo)
                    os.rmdir(caminho_item)
                    self.logger.info(f"Folder {item} removed.")

        self.logger.info("Process completed.")

    def reads_and_concatenates_csv_files(self) -> DataFrame:
        """
        Reads and concatenates files.

        Returns
        ----------
        df_final: DataFrame
            Final DataFrame.
        """

        dfs = [
            pd.read_csv(
                os.path.join(self.data_dir, file),
                usecols=[
                    "CD_ORGAO",
                    "NM_ORGAO",
                    "NR_LICITACAO",
                    "ANO_LICITACAO",
                    "CD_TIPO_MODALIDADE",
                    "NR_COMISSAO",
                    "TP_OBJETO",
                    "CD_TIPO_FASE_ATUAL",
                    "TP_LICITACAO",
                    "TP_NIVEL_JULGAMENTO",
                    "TP_CARACTERISTICA_OBJETO",
                    "TP_NATUREZA",
                    "DS_OBJETO",
                    "VL_LICITACAO",
                    "BL_PERMITE_CONSORCIO",
                    "DT_ABERTURA",
                    "DT_HOMOLOGACAO",
                    "DT_ADJUDICACAO",
                    "BL_LICIT_PROPRIA_ORGAO",
                    "VL_HOMOLOGADO",
                    "DS_OBSERVACAO",
                ],
            )
            for file in os.listdir(self.data_dir)
            if file.endswith(".csv")
        ]

        df_final = pd.concat(dfs, ignore_index=True)   
        
        # Getting only those bids where `CD_TIPO_FASE_ATUAL=ADH`,
        # which are the ones that have been approved,
        # according to the TCE documentation in page 27.
             
        df_final = df_final[df_final["CD_TIPO_FASE_ATUAL"] == "ADH"]
        
        return df_final

    def saves_df_to_csv(self, df_final: DataFrame):
        """
        Saves DataFrame to csv.

        Parameters
        ----------
        df_final: DataFrame
            Final DataFrame.
        """

        os.system(f"rm -rf data/*")

        df_final.to_csv(f"{self.data_dir}/tce_licitations.csv", index=False)

    def get(self):
        """Runs the full process"""

        self.downloads_data()
        self.extract_files()
        self.rename_csv_files()
        self.delete_non_licitation_files()
        self.cleanup()
        df_final = self.reads_and_concatenates_csv_files()
        self.saves_df_to_csv(df_final)

        self.logger.info(f"TCE data saved to {self.data_dir} folder!")


if __name__ == "__main__":
    tce_data = DataTCE(data_dir="data", logger=logger)
    tce_data.get()
