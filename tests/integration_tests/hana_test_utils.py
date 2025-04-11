"""Helper utilities for HANA integration tests."""

from datetime import datetime, timedelta


class HanaTestUtils:
    @staticmethod
    def execute_sql(conn, statement, parameters=None, return_result=False):
        assert conn, "Connection cannot be None"
        cursor = conn.cursor()
        assert cursor
        res = None
        if parameters is not None:
            cursor.executemany(statement, parameters)
        else:
            cursor.execute(statement)
        if return_result:
            res = cursor.fetchone()
        cursor.close()
        conn.commit()
        if return_result:
            return res[0]

    @staticmethod
    def drop_schema_if_exists(conn, schema_name):
        res = HanaTestUtils.execute_sql(
            conn,
            f"SELECT COUNT(*) FROM SYS.SCHEMAS WHERE SCHEMA_NAME='{schema_name}'",
            return_result=True,
        )
        if res != 0:
            HanaTestUtils.execute_sql(conn, f'DROP SCHEMA "{schema_name}" CASCADE')

    @staticmethod
    def drop_old_test_schemas(conn, schema_prefix):
        try:
            assert conn
            cursor = conn.cursor()
            assert cursor
            sql = f"""SELECT SCHEMA_NAME FROM SYS.SCHEMAS WHERE SCHEMA_NAME
                      LIKE '{schema_prefix.replace('_', '__')}__%' ESCAPE '_' AND
                      LOCALTOUTC(CREATE_TIME) < ?"""
            cursor.execute(sql, datetime.now() - timedelta(days=1))
            rows = cursor.fetchall()

            for row in rows:
                HanaTestUtils.execute_sql(
                    conn, f'DROP SCHEMA "{row["SCHEMA_NAME"]}" CASCADE'
                )
        except Exception as ex:
            print(f"Unable to drop old test schemas. Error: {ex}")
            pass
        finally:
            cursor.close()

    @staticmethod
    def generate_schema_name(conn, prefix):
        sql = (
            "SELECT REPLACE(CURRENT_UTCDATE, '-', '') || '_' || BINTOHEX(SYSUUID) "
            "FROM DUMMY;"
        )
        uid = HanaTestUtils.execute_sql(conn, sql, return_result=True)
        return f"{prefix}_{uid}"

    @staticmethod
    def create_and_set_schema(conn, schema_name):
        # HanaTestUtils.dropSchemaIfExists(conn, schema_name)
        HanaTestUtils.execute_sql(conn, f'CREATE SCHEMA "{schema_name}"')
        HanaTestUtils.execute_sql(conn, f'SET SCHEMA "{schema_name}"')
