import os
import platform
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import sqlite_vec
from pandas.testing import assert_frame_equal

from feast import FeatureStore, RepoConfig
from feast.errors import FeatureViewNotFoundException
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import FloatList as FloatListProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import RegistryConfig
from tests.integration.feature_repos.universal.feature_views import TAGS
from tests.utils.cli_repo_creator import CliRunner, get_example_repo


def test_get_online_features() -> None:
    """
    Test reading from the online store in local mode.
    """
    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"), "file"
    ) as store:
        # Write some data to two tables
        driver_locations_fv = store.get_feature_view(name="driver_locations")
        customer_profile_fv = store.get_feature_view(name="customer_profile")
        customer_driver_combined_fv = store.get_feature_view(
            name="customer_driver_combined"
        )

        provider = store._get_provider()

        driver_key = EntityKeyProto(
            join_keys=["driver_id"], entity_values=[ValueProto(int64_val=1)]
        )
        provider.online_write_batch(
            config=store.config,
            table=driver_locations_fv,
            data=[
                (
                    driver_key,
                    {
                        "lat": ValueProto(double_val=0.1),
                        "lon": ValueProto(string_val="1.0"),
                    },
                    datetime.utcnow(),
                    datetime.utcnow(),
                )
            ],
            progress=None,
        )

        customer_key = EntityKeyProto(
            join_keys=["customer_id"], entity_values=[ValueProto(string_val="5")]
        )
        provider.online_write_batch(
            config=store.config,
            table=customer_profile_fv,
            data=[
                (
                    customer_key,
                    {
                        "avg_orders_day": ValueProto(float_val=1.0),
                        "name": ValueProto(string_val="John"),
                        "age": ValueProto(int64_val=3),
                    },
                    datetime.utcnow(),
                    datetime.utcnow(),
                )
            ],
            progress=None,
        )

        customer_key = EntityKeyProto(
            join_keys=["customer_id", "driver_id"],
            entity_values=[ValueProto(string_val="5"), ValueProto(int64_val=1)],
        )
        provider.online_write_batch(
            config=store.config,
            table=customer_driver_combined_fv,
            data=[
                (
                    customer_key,
                    {"trips": ValueProto(int64_val=7)},
                    datetime.utcnow(),
                    datetime.utcnow(),
                )
            ],
            progress=None,
        )

        assert len(store.list_entities()) == 3
        assert len(store.list_entities(tags=TAGS)) == 2

        # Retrieve two features using two keys, one valid one non-existing
        result = store.get_online_features(
            features=[
                "driver_locations:lon",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_driver_combined:trips",
            ],
            entity_rows=[
                {"driver_id": 1, "customer_id": "5"},
                {"driver_id": 1, "customer_id": 5},
            ],
            full_feature_names=False,
        ).to_dict()

        assert "lon" in result
        assert "avg_orders_day" in result
        assert "name" in result
        assert result["driver_id"] == [1, 1]
        assert result["customer_id"] == ["5", "5"]
        assert result["lon"] == ["1.0", "1.0"]
        assert result["avg_orders_day"] == [1.0, 1.0]
        assert result["name"] == ["John", "John"]
        assert result["trips"] == [7, 7]

        # Ensure features are still in result when keys not found
        result = store.get_online_features(
            features=["customer_driver_combined:trips"],
            entity_rows=[{"driver_id": 0, "customer_id": 0}],
            full_feature_names=False,
        ).to_dict()

        assert "trips" in result

        result = store.get_online_features(
            features=["customer_profile_pandas_odfv:on_demand_age"],
            entity_rows=[{"driver_id": 1, "customer_id": "5"}],
            full_feature_names=True,
        ).to_dict()

        assert "on_demand_age" in [i.split("__")[-1] for i in result]
        assert result["driver_id"] == [1]
        assert result["customer_id"] == ["5"]
        assert result["customer_profile_pandas_odfv__on_demand_age"] == [4]

        # invalid table reference
        with pytest.raises(FeatureViewNotFoundException):
            store.get_online_features(
                features=["driver_locations_bad:lon"],
                entity_rows=[{"driver_id": 1}],
                full_feature_names=False,
            )

        # Create new FeatureStore object with fast cache invalidation
        cache_ttl = 1
        fs_fast_ttl = FeatureStore(
            config=RepoConfig(
                registry=RegistryConfig(
                    path=store.config.registry.path, cache_ttl_seconds=cache_ttl
                ),
                online_store=store.config.online_store,
                project=store.project,
                provider=store.config.provider,
                entity_key_serialization_version=2,
            )
        )

        # Should download the registry and cache it permanently (or until manually refreshed)
        result = fs_fast_ttl.get_online_features(
            features=[
                "driver_locations:lon",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_driver_combined:trips",
            ],
            entity_rows=[{"driver_id": 1, "customer_id": 5}],
            full_feature_names=False,
        ).to_dict()
        assert result["lon"] == ["1.0"]
        assert result["trips"] == [7]

        # Rename the registry.db so that it cant be used for refreshes
        os.rename(store.config.registry.path, store.config.registry.path + "_fake")

        # Wait for registry to expire
        time.sleep(cache_ttl)

        # Will try to reload registry because it has expired (it will fail because we deleted the actual registry file)
        with pytest.raises(FileNotFoundError):
            fs_fast_ttl.get_online_features(
                features=[
                    "driver_locations:lon",
                    "customer_profile:avg_orders_day",
                    "customer_profile:name",
                    "customer_driver_combined:trips",
                ],
                entity_rows=[{"driver_id": 1, "customer_id": 5}],
                full_feature_names=False,
            ).to_dict()

        # Restore registry.db so that we can see if it actually reloads registry
        os.rename(store.config.registry.path + "_fake", store.config.registry.path)

        # Test if registry is actually reloaded and whether results return
        result = fs_fast_ttl.get_online_features(
            features=[
                "driver_locations:lon",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_driver_combined:trips",
            ],
            entity_rows=[{"driver_id": 1, "customer_id": 5}],
            full_feature_names=False,
        ).to_dict()
        assert result["lon"] == ["1.0"]
        assert result["trips"] == [7]

        # Create a registry with infinite cache (for users that want to manually refresh the registry)
        fs_infinite_ttl = FeatureStore(
            config=RepoConfig(
                registry=RegistryConfig(
                    path=store.config.registry.path, cache_ttl_seconds=0
                ),
                online_store=store.config.online_store,
                project=store.project,
                provider=store.config.provider,
                entity_key_serialization_version=2,
            )
        )

        # Should return results (and fill the registry cache)
        result = fs_infinite_ttl.get_online_features(
            features=[
                "driver_locations:lon",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_driver_combined:trips",
            ],
            entity_rows=[{"driver_id": 1, "customer_id": 5}],
            full_feature_names=False,
        ).to_dict()
        assert result["lon"] == ["1.0"]
        assert result["trips"] == [7]

        # Wait a bit so that an arbitrary TTL would take effect
        time.sleep(2)

        # Rename the registry.db so that it cant be used for refreshes
        os.rename(store.config.registry.path, store.config.registry.path + "_fake")

        # TTL is infinite so this method should use registry cache
        result = fs_infinite_ttl.get_online_features(
            features=[
                "driver_locations:lon",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_driver_combined:trips",
            ],
            entity_rows=[{"driver_id": 1, "customer_id": 5}],
            full_feature_names=False,
        ).to_dict()
        assert result["lon"] == ["1.0"]
        assert result["trips"] == [7]

        # Force registry reload (should fail because file is missing)
        with pytest.raises(FileNotFoundError):
            fs_infinite_ttl.refresh_registry()

        # Restore registry.db so that teardown works
        os.rename(store.config.registry.path + "_fake", store.config.registry.path)


def test_online_to_df():
    """
    Test dataframe conversion. Make sure the response columns and rows are
    the same order as the request.
    """
    driver_ids = [1, 2, 3]
    customer_ids = [4, 5, 6]
    name = "foo"
    lon_multiply = 1.0
    lat_multiply = 0.1
    age_multiply = 10
    avg_order_day_multiply = 1.0

    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"), "file"
    ) as store:
        # Write three tables to online store
        driver_locations_fv = store.get_feature_view(name="driver_locations")
        customer_profile_fv = store.get_feature_view(name="customer_profile")
        customer_driver_combined_fv = store.get_feature_view(
            name="customer_driver_combined"
        )
        provider = store._get_provider()

        for d, c in zip(driver_ids, customer_ids):
            """
            driver table:
                                    lon                    lat
                1                   1.0                    0.1
                2                   2.0                    0.2
                3                   3.0                    0.3
            """
            driver_key = EntityKeyProto(
                join_keys=["driver_id"], entity_values=[ValueProto(int64_val=d)]
            )
            provider.online_write_batch(
                config=store.config,
                table=driver_locations_fv,
                data=[
                    (
                        driver_key,
                        {
                            "lat": ValueProto(double_val=d * lat_multiply),
                            "lon": ValueProto(string_val=str(d * lon_multiply)),
                        },
                        datetime.utcnow(),
                        datetime.utcnow(),
                    )
                ],
                progress=None,
            )

            """
            customer table
            customer     avg_orders_day          name        age
                4           4.0                  foo4         40
                5           5.0                  foo5         50
                6           6.0                  foo6         60
            """
            customer_key = EntityKeyProto(
                join_keys=["customer_id"], entity_values=[ValueProto(string_val=str(c))]
            )
            provider.online_write_batch(
                config=store.config,
                table=customer_profile_fv,
                data=[
                    (
                        customer_key,
                        {
                            "avg_orders_day": ValueProto(
                                float_val=c * avg_order_day_multiply
                            ),
                            "name": ValueProto(string_val=name + str(c)),
                            "age": ValueProto(int64_val=c * age_multiply),
                        },
                        datetime.utcnow(),
                        datetime.utcnow(),
                    )
                ],
                progress=None,
            )
            """
            customer_driver_combined table
            customer  driver    trips
                4       1       4
                5       2       10
                6       3       18
            """
            combo_keys = EntityKeyProto(
                join_keys=["customer_id", "driver_id"],
                entity_values=[ValueProto(string_val=str(c)), ValueProto(int64_val=d)],
            )
            provider.online_write_batch(
                config=store.config,
                table=customer_driver_combined_fv,
                data=[
                    (
                        combo_keys,
                        {"trips": ValueProto(int64_val=c * d)},
                        datetime.utcnow(),
                        datetime.utcnow(),
                    )
                ],
                progress=None,
            )

        # Get online features in dataframe
        result_df = store.get_online_features(
            features=[
                "driver_locations:lon",
                "driver_locations:lat",
                "customer_profile:avg_orders_day",
                "customer_profile:name",
                "customer_profile:age",
                "customer_driver_combined:trips",
            ],
            # Reverse the row order
            entity_rows=[
                {"driver_id": d, "customer_id": c}
                for (d, c) in zip(reversed(driver_ids), reversed(customer_ids))
            ],
        ).to_df()
        """
        Construct the expected dataframe with reversed row order like so:
        driver  customer     lon    lat     avg_orders_day      name        age     trips
            3       6        3.0    0.3         6.0             foo6        60       18
            2       5        2.0    0.2         5.0             foo5        50       10
            1       4        1.0    0.1         4.0             foo4        40       4
        """
        df_dict = {
            "driver_id": driver_ids,
            "customer_id": [str(c) for c in customer_ids],
            "lon": [str(d * lon_multiply) for d in driver_ids],
            "lat": [d * lat_multiply for d in driver_ids],
            "avg_orders_day": [c * avg_order_day_multiply for c in customer_ids],
            "name": [name + str(c) for c in customer_ids],
            "age": [c * age_multiply for c in customer_ids],
            "trips": [d * c for (d, c) in zip(driver_ids, customer_ids)],
        }
        # Requested column order
        ordered_column = [
            "driver_id",
            "customer_id",
            "lon",
            "lat",
            "avg_orders_day",
            "name",
            "age",
            "trips",
        ]
        expected_df = pd.DataFrame({k: reversed(v) for (k, v) in df_dict.items()})
        assert_frame_equal(result_df[ordered_column], expected_df)


@pytest.mark.skipif(
    sys.version_info[0:2] != (3, 10) or platform.system() != "Darwin",
    reason="Only works on Python 3.10 and MacOS",
)
def test_sqlite_get_online_documents() -> None:
    """
    Test retrieving documents from the online store in local mode.
    """
    n = 10  # number of samples - note: we'll actually double it
    vector_length = 8
    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"), "file"
    ) as store:
        store.config.online_store.vec_enabled = True
        store.config.online_store.vector_len = vector_length
        # Write some data to two tables
        document_embeddings_fv = store.get_feature_view(name="document_embeddings")

        provider = store._get_provider()

        item_keys = [
            EntityKeyProto(
                join_keys=["item_id"], entity_values=[ValueProto(int64_val=i)]
            )
            for i in range(n)
        ]
        data = []
        for item_key in item_keys:
            data.append(
                (
                    item_key,
                    {
                        "Embeddings": ValueProto(
                            float_list_val=FloatListProto(
                                val=np.random.random(
                                    vector_length,
                                )
                            )
                        )
                    },
                    datetime.utcnow(),
                    datetime.utcnow(),
                )
            )

        provider.online_write_batch(
            config=store.config,
            table=document_embeddings_fv,
            data=data,
            progress=None,
        )
        documents_df = pd.DataFrame(
            {
                "item_id": [str(i) for i in range(n)],
                "Embeddings": [
                    np.random.random(
                        vector_length,
                    )
                    for i in range(n)
                ],
                "event_timestamp": [datetime.utcnow() for _ in range(n)],
            }
        )

        store.write_to_online_store(
            feature_view_name="document_embeddings",
            df=documents_df,
        )

        document_table = store._provider._online_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' and name like '%_document_embeddings';"
        ).fetchall()
        assert len(document_table) == 1
        document_table_name = document_table[0][0]
        record_count = len(
            store._provider._online_store._conn.execute(
                f"select * from {document_table_name}"
            ).fetchall()
        )
        assert record_count == len(data) + documents_df.shape[0]

        query_embedding = np.random.random(
            vector_length,
        )
        result = store.retrieve_online_documents(
            feature="document_embeddings:Embeddings", query=query_embedding, top_k=3
        ).to_dict()

        assert "Embeddings" in result
        assert "distance" in result
        assert len(result["distance"]) == 3


@pytest.mark.skipif(
    sys.version_info[0:2] != (3, 10) or platform.system() != "Darwin",
    reason="Only works on Python 3.10 and MacOS",
)
def test_sqlite_vec_import() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    db.execute("""
    create virtual table vec_examples using vec0(
      sample_embedding float[8]
    );
    """)

    db.execute("""
    insert into vec_examples(rowid, sample_embedding)
    values
        (1, '[-0.200, 0.250, 0.341, -0.211, 0.645, 0.935, -0.316, -0.924]'),
        (2, '[0.443, -0.501, 0.355, -0.771, 0.707, -0.708, -0.185, 0.362]'),
        (3, '[0.716, -0.927, 0.134, 0.052, -0.669, 0.793, -0.634, -0.162]'),
        (4, '[-0.710, 0.330, 0.656, 0.041, -0.990, 0.726, 0.385, -0.958]');
    """)

    sqlite_version, vec_version = db.execute(
        "select sqlite_version(), vec_version()"
    ).fetchone()
    assert vec_version == "v0.0.1-alpha.10"
    print(f"sqlite_version={sqlite_version}, vec_version={vec_version}")

    result = db.execute("""
        select
            rowid,
            distance
        from vec_examples
        where sample_embedding match '[0.890, 0.544, 0.825, 0.961, 0.358, 0.0196, 0.521, 0.175]'
        order by distance
        limit 2;
    """).fetchall()
    result = [(rowid, round(distance, 2)) for rowid, distance in result]
    assert result == [(2, 2.39), (1, 2.39)]
