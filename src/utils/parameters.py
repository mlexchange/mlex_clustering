from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class KmeansParameters(BaseModel):
    n_clusters: int = Field(description="number of clusters")


class DBSCANParameters(BaseModel):
    eps: float = Field(description="max distance between two samples")
    min_samples: int = Field(description="number of samples in a neighborhood")


class HDBSCANParameters(BaseModel):
    min_cluster_size: int = Field(description="minimum number of samples in a cluster")


class DataType(str, Enum):
    tiled = "tiled"


class IOParameters(BaseModel):
    uid_retrieve: str = Field(description="uid to retrieve data from")
    data_type: DataType = Field(description="data type")
    root_uri: str = Field(description="root uri")
    data_uris: list[str] = Field(description="data uris")
    data_tiled_api_key: Optional[str] = Field(description="tiled api key")
    results_tiled_uri: str = Field(description="tiled uri to save data to")
    results_tiled_api_key: Optional[str] = Field(description="tiled api key")
    uid_save: str = Field(description="uid to save data to")
    results_dir: str = Field(description="results directory")
