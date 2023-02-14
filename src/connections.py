"""Module for connecting to the MySQL database."""

import pymysql
import pymysql.cursors

from src.config import DB, HOST, PASSWORD, USER


def get_mysql_connection():
    """Return a connection to the MySQL database.

    Returns:
        pymysql.connections.Connection: a connection to the MySQL database.
    """
    if not all([HOST, USER, PASSWORD, DB]):
        raise ValueError(
            "No MySQL database configuration found. Add environment variables to .env file."
        )
    return pymysql.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,  # type: ignore
        db=DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
