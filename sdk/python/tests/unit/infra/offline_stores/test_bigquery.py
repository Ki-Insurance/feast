from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow
import pytest

from feast.infra.offline_stores.bigquery import (
    BigQueryOfflineStore,
    BigQueryOfflineStoreConfig,
    BigQueryRetrievalJob,
)
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RepoConfig


@pytest.fixture
def pandas_dataframe():
    return pd.DataFrame(
        data={
            "key": [1, 2, 3],
            "value": ["a", None, "c"],
        }
    )


@pytest.fixture
def big_query_result(pandas_dataframe):
    class BigQueryResult:
        def to_dataframe(self, **kwargs):
            return pandas_dataframe

        def to_arrow(self, **kwargs):
            return pyarrow.Table.from_pandas(pandas_dataframe)

        def exception(self, timeout=None):
            return None

    return BigQueryResult()


class TestBigQueryRetrievalJob:
    query = "SELECT * FROM bigquery"
    client = Mock()
    retrieval_job = BigQueryRetrievalJob(
        query=query,
        client=client,
        config=RepoConfig(
            registry="gs://ml-test/repo/registry.db",
            project="test",
            provider="gcp",
            online_store=SqliteOnlineStoreConfig(type="sqlite"),
            offline_store=BigQueryOfflineStoreConfig(type="bigquery", dataset="feast"),
        ),
        full_feature_names=True,
        on_demand_feature_views=[],
    )

    def test_to_sql(self):
        assert self.retrieval_job.to_sql() == self.query

    def test_to_df(self, big_query_result, pandas_dataframe):
        self.client.query.return_value = big_query_result
        actual = self.retrieval_job.to_df()
        pd.testing.assert_frame_equal(actual, pandas_dataframe)

    def test_to_df_timeout(self, big_query_result):
        self.client.query.return_value = big_query_result
        with patch.object(self.retrieval_job, "_execute_query"):
            self.retrieval_job.to_df(timeout=30)
            self.retrieval_job._execute_query.assert_called_once_with(
                query=self.query, timeout=30
            )

    def test_to_arrow(self, big_query_result, pandas_dataframe):
        self.client.query.return_value = big_query_result
        actual = self.retrieval_job.to_arrow()
        pd.testing.assert_frame_equal(actual.to_pandas(), pandas_dataframe)

    def test_to_arrow_timeout(self, big_query_result):
        self.client.query.return_value = big_query_result
        with patch.object(self.retrieval_job, "_execute_query"):
            self.retrieval_job.to_arrow(timeout=30)
            self.retrieval_job._execute_query.assert_called_once_with(
                query=self.query, timeout=30
            )


class TestOfflineWriteBatch:
    @patch("feast.infra.offline_stores.bigquery._get_bigquery_client")
    def test_offline_write_batch_enables_list_inference(self, mock_get_client):
        """LoadJobConfig must set parquet_options.enable_list_inference = True
        so that BigQuery correctly interprets PyArrow list columns from parquet.

        Without it BigQuery reads the parquet 3-level LIST wrapper as a RECORD,
        and reconciles it against the REPEATED schema Feast supplies by writing
        the row with every array empty — no error, no bad records. See MOPS-1266.
        """
        source = BigQuerySource(
            name="test",
            table="project.dataset.table",
            timestamp_field="ts",
        )
        fv = MagicMock()
        fv.batch_source = source

        pa_schema = pyarrow.schema(
            [
                pyarrow.field("entity_id", pyarrow.string()),
                pyarrow.field("tags", pyarrow.list_(pyarrow.string())),
                pyarrow.field("ts", pyarrow.timestamp("us", tz="UTC")),
            ]
        )
        pa_table = pyarrow.table(
            {
                "entity_id": ["e1"],
                "tags": [["a", "b"]],
                "ts": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            },
            schema=pa_schema,
        )

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.load_table_from_file.return_value = MagicMock()

        config = RepoConfig(
            registry="gs://test/registry.db",
            project="test",
            provider="gcp",
            offline_store=BigQueryOfflineStoreConfig(project_id="test-project"),
            online_store=SqliteOnlineStoreConfig(),
        )

        with patch(
            "feast.infra.offline_stores.offline_utils.get_pyarrow_schema_from_batch_source",
            return_value=(pa_schema, pa_table.column_names),
        ):
            BigQueryOfflineStore.offline_write_batch(
                config=config,
                feature_view=fv,
                table=pa_table,
                progress=None,
            )

        call_kwargs = mock_client.load_table_from_file.call_args
        job_config = call_kwargs[1]["job_config"]
        assert job_config.parquet_options is not None
        assert job_config.parquet_options.enable_list_inference is True
