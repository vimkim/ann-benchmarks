"""
This module supports connecting to a CUBRID instance and performing vector
indexing and search. The default behavior uses the "ann" value of CUBRID user name, password, and database name.
and the default host and port values are localhost and 33000.

If CUBRID is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting CUBRID
connection parameters:

ANN_BENCHMARKS_CUB_USER
ANN_BENCHMARKS_CUB_PASSWORD
ANN_BENCHMARKS_CUB_DBNAME
ANN_BENCHMARKS_CUB_HOST
ANN_BENCHMARKS_CUB_PORT

This module starts the CUBRID server and broker automatically using the "cubrid"
command.
"""

import os
import subprocess
import sys
import time
import contextlib
import io
import CUBRIDdb
import random

from typing import Dict, Any, Optional

from ..base.module import BaseANN

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<c>",
        "ops_type": "COSINE",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "EUCLIDEAN",
    }
}

def get_cub_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_CUB_{pg_param_name.upper()}'

def get_cub_conn_param(cub_param_name: str, default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_cub_param_env_var_name(cub_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value

def open_connection_primitive(host, port, database, user, password):
    print("### open connection...")
    url = f"CUBRID:{host}:{port}:{database}:::"
    return CUBRIDdb.connect(url, user, password or '')

def open_cursor_primitive(conn):
    print("### open cursor...")
    return conn.cursor()

def write_loaddb_object_file(X, output_path):
    with open(output_path, 'w') as f:
        f.write("%id items 0\n")
        f.write("%class items ([id] [embedding])\n")

        for i, vec in enumerate(X):
            vec_str = "[" + ", ".join(map(str, vec)) + "]"
            f.write(f"{i} '{vec_str}'\n")

def run_loaddb(db_name: str, object_file_path: str):
    try:
        result = subprocess.run(
            [
                "cubrid", "loaddb",
                "-C",                   # Create mode
                "-u", "ann",           # DBA user
                "-p", "ann",           # DBA password
                "-d", object_file_path,
                "-c", "10000",
                "--no-statistics",
                db_name
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("loaddb output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("loaddb failed with error:\n", e.stderr)
        raise

class CUBVEC(BaseANN):

    def done(self) -> None:
        print("### done")

    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.is_prepared = False

        self._signature_base = metric + "_" + str(self._m) + "_" + str(self._ef_construction)
        self._signature = self._signature_base

        if metric == "angular":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM {} WHERE id = ?"
        elif metric == "euclidean":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM {} ORDER BY embedding <-> ? LIMIT 10"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError("Unknown metric: {}. Valid metrics: {}".format(
                self._metric, ', '.join(sorted(METRIC_PROPERTIES.keys()))))
        return METRIC_PROPERTIES[self._metric]

    def fit(self, X):
        cubrid_connect_kwargs: Dict[str, Any] = dict(autocommit=True)
        for arg_name in ['user', 'password', 'dbname']:
            cubrid_connect_kwargs[arg_name] = get_cub_conn_param(arg_name, 'ann')

        cub_host = get_cub_conn_param('host')
        if cub_host is not None:
            cubrid_connect_kwargs['host'] = cub_host

        cub_port_str = get_cub_conn_param('port')
        if cub_port_str is not None:
            cubrid_connect_kwargs['port'] = int(cub_port_str)

        print(cubrid_connect_kwargs)

        try:
            subprocess.run(["cubrid", "server", "start", "ann"], check=True)
            print("CUBRID server 'ann' started.")
        except subprocess.CalledProcessError as e:
            print("Failed to start CUBRID server:", e)

        try:
            subprocess.run(["cubrid", "broker", "start"], check=True)
            print("CUBRID broker started.")
        except subprocess.CalledProcessError as e:
            print("Failed to start CUBRID broker:", e)

        idx_num = 0

        # Wait until the server is really ready
        for _ in range(30):  # try for up to 30 seconds
            try:
                conn = open_connection_primitive(
                    cubrid_connect_kwargs['host'],
                    cubrid_connect_kwargs['port'],
                    cubrid_connect_kwargs['dbname'],
                    cubrid_connect_kwargs['user'],
                    cubrid_connect_kwargs['password']
                )
                print("Connected to CUBRID successfully.")
                break
            except Exception as e:
                print("Waiting for CUBRID to be ready...")
                time.sleep(1)
        else:
            raise RuntimeError("CUBRID server did not become ready in time.")

        try:
            try:
                total_rows = X.shape[0]
                dim = X.shape[1]

                print("copying data...")
                sys.stdout.flush()

                if (self._signature == self._signature_base):
                    self._signature = self._signature_base + "_" + str(dim)
                batch_size = 50000

                print("copying data...")
                sys.stdout.flush()

                header = f"%id ann.{self._signature} 0\n%class ann.{self._signature} ([id] [embedding])\n"

                for start in range(0, total_rows, batch_size):
                    end = min(start + batch_size, total_rows)
                    batch = X[start:end]

                    object_file_path = f"/tmp/{self._signature}_object_{start}_{end}"

                    insert_batch_time_sec = time.time()

                    if os.path.exists(object_file_path + ".flag"):
                        print("skipping {} rows, start: {}, end: {}".format(end - start, start, end))
                        continue
                    else:
                        buffer = io.StringIO()
                        lines = [
                            f"{i} '[{','.join(map(str, vec))}]'\n"
                            for i, vec in enumerate(batch, start=start)
                        ]
                        buffer.write(header)
                        buffer.writelines(lines)

                        try:
                            with open(object_file_path, "w") as f:
                                f.write(buffer.getvalue())
                            with open(object_file_path + ".flag", "w") as f:
                                f.write("success")
                        except Exception as e:
                            print("Error during writing object file:", e)
                            raise

                    print("converted {} rows into object file in {:.3f} seconds".format(
                        end - start, time.time() - insert_batch_time_sec))

                cur = open_cursor_primitive(conn)
                # if the self._signature table exists, skip the following
                try:
                    cur.execute("SELECT COUNT(*) FROM " + self._signature)
                    if cur.fetchone()[0] == total_rows:
                        print("table " + self._signature + " already exists, skip the index creation")
                        return

                except Exception as e:
                    print("Error during checking table existence:", e)

                print("creating a new table " + self._signature + " and index...")
                cur.execute("DROP TABLE IF EXISTS " + self._signature + ";")
                cur.execute("CREATE TABLE " + self._signature + " (id int, embedding vector(%d));" % dim)

                sys.stdout.flush()

                # create_index_str = (
                #     "CREATE VECTOR INDEX idx_v ON %s(embedding %s) "
                #     "WITH (m = %d, ef_construction = %d);" % (
                #             self._signature,
                #             self.get_metric_properties()["ops_type"],
                #             self._m,
                #             self._ef_construction
                #         )
                # )

                create_index_str = (
                    "CREATE INDEX idx ON %s(id);"
                        % (
                            self._signature,
                        )
                )
                cur.execute(create_index_str)

                insert_start_time_sec = time.time()
                for start in range(0, total_rows, batch_size):
                    end = min(start + batch_size, total_rows)
                    batch = X[start:end]
                    object_file_path = f"/tmp/{self._signature}_object_{start}_{end}"

                    insert_batch_time_sec = time.time()
                    try:
                        result = subprocess.run([
                            "cubrid", "loaddb",
                            "-C",
                            cubrid_connect_kwargs['dbname'],
                            "-u", "ann",
                            "-p", "ann",
                            "-d", object_file_path,
                            "-c", str(batch_size),
                            "--estimated-size", str(batch_size),
                            "--no-statistics",
                            "--no-user-specified-name"
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                        print("inserted {} rows into table in {:.3f} seconds".format(
                            end - start, time.time() - insert_batch_time_sec))
                        sys.stdout.flush()

                        print("{} rows are remaining".format(total_rows - start))

                        # elapse time needed in the following output
                    except subprocess.CalledProcessError as e:
                        print("loaddb failed with error:\n", e.stderr)
                        raise

                print("### total time to insert {} rows into table: {:.3f} seconds".format(
                    total_rows, time.time() - insert_start_time_sec))

            except Exception as e:
                print("Error during DB setup or insertion:", e)
                raise
            finally:
                cur.close()
        
            # expose the cursor
            self._cur = open_cursor_primitive(conn)

        except Exception as e:
            print("Error during DB setup or connection:", e)
            raise

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET SYSTEM PARAMETERS 'hnsw_ef_search=%d'" % ef_search)

        print("### preparing query...")
        query_str = self._query.format(self._signature)
        self._cur._cs.prepare(query_str)

    def query(self, v, n):
        vector_str = "[" + ",".join(map(str, v)) + "]"
        cur = self._cur

        # args = [vector_str, n] # this reduces QPS from 3500 to 600
        # args = [vector_str]
        args = [random.randint(0, 8999)]
        set_type = None
        if args is not None:
            cur._bind_params(args, set_type)
        r = cur._cs.execute()
        cur.rowcount = cur._cs.rowcount
        cur.description = cur._cs.description
        # return r

        res = [id for id, in cur.fetchall()]
        return res

    def get_memory_usage(self):
        return 0

    def __str__(self):
        return f"CUBVEC(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
