from tiled.client import from_uri
from tiled.structures.data_source import Asset, DataSource
from tiled.structures.table import TableStructure


class TiledDataset:
    def __init__(
        self, read_tiled_uri, write_tiled_uri, read_tiled_key=None, write_tiled_key=None
    ):
        if read_tiled_uri:
            self.read_client = from_uri(read_tiled_uri, api_key=read_tiled_key)
        self.write_client = from_uri(write_tiled_uri, api_key=write_tiled_key)

    def load_data_from_tiled(self, uri):
        return self.read_client[uri].read().to_numpy()

    def write_results(self, labels, io_parameters, labels_path, metadata=None):
        # Prepare Tiled parent node
        uid_save = io_parameters.uid_save

        # Save latent vectors to Tiled
        structure = TableStructure.from_pandas(labels)

        # Remove API keys from metadata
        if metadata:
            metadata["io_parameters"].pop("data_tiled_api_key", None)
            metadata["io_parameters"].pop("results_tiled_api_key", None)

        frame = self.write_client.new(
            structure_family="table",
            data_sources=[
                DataSource(
                    structure_family="table",
                    structure=structure,
                    mimetype="application/x-parquet",
                    assets=[
                        Asset(
                            data_uri=f"file://{labels_path}",
                            is_directory=False,
                            parameter="data_uris",
                            num=1,
                        )
                    ],
                )
            ],
            metadata=metadata,
            key=uid_save,
        )

        frame.write(labels)
        pass
