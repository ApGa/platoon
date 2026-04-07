from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass

from platoon.email_search.data.local_email_db import DB_PATH_ENV_VAR, resolve_db_path
from platoon.email_search.data.types_enron import Email

logger = logging.getLogger(__name__)
_CONN: sqlite3.Connection | None = None
_CONN_DB_PATH: str | None = None


def get_conn() -> sqlite3.Connection:
    global _CONN, _CONN_DB_PATH
    db_path = resolve_db_path()
    if _CONN is None or _CONN_DB_PATH != db_path:
        if _CONN is not None:
            _CONN.close()
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Email database not found at {db_path}. "
                "Generate it with "
                f"`python -m platoon.email_search.data.local_email_db --db-path {db_path} --overwrite` "
                f"or set `{DB_PATH_ENV_VAR}`."
            )
        _CONN = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        _CONN_DB_PATH = db_path
    return _CONN


@dataclass
class SearchResult:
    message_id: str
    snippet: str


def _search_emails_sync(
    inbox: str,
    keywords: list[str],
    from_addr: str | None = None,
    to_addr: str | None = None,
    sent_after: str | None = None,
    sent_before: str | None = None,
    max_results: int = 10,
) -> list[SearchResult]:
    if not keywords:
        raise ValueError("No keywords provided for search.")
    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    cursor = get_conn().cursor()
    where_clauses: list[str] = []
    params: list[str | int] = []

    escaped_keywords = [keyword.replace('"', '""') for keyword in keywords]
    fts_query = " ".join(f'"{keyword}"' for keyword in escaped_keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
        """
    )
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
            """
        )
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
    SELECT
        e.message_id,
        snippet(emails_fts, -1, ' ', ' ', ' ... ', 15) as snippet
    FROM
        emails e JOIN emails_fts fts ON e.id = fts.rowid
    WHERE
        {" AND ".join(where_clauses)}
    ORDER BY
        e.date DESC
    LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = [SearchResult(message_id=row[0], snippet=row[1]) for row in cursor.fetchall()]
    logger.info("Search found %s results for inbox=%s", len(results), inbox)
    return results


async def search_emails(
    inbox: str,
    keywords: list[str],
    from_addr: str | None = None,
    to_addr: str | None = None,
    sent_after: str | None = None,
    sent_before: str | None = None,
    max_results: int = 10,
) -> list[SearchResult]:
    return await asyncio.to_thread(
        _search_emails_sync,
        inbox,
        keywords,
        from_addr,
        to_addr,
        sent_after,
        sent_before,
        max_results,
    )


def _read_email_sync(message_id: str) -> Email | None:
    cursor = get_conn().cursor()
    cursor.execute(
        """
        SELECT id, message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
        """,
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        logger.warning("Email with message_id '%s' not found.", message_id)
        return None

    email_pk_id, msg_id, date, subject, from_addr, body, file_name = email_row
    cursor.execute(
        """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
        """,
        (email_pk_id,),
    )

    to_addresses: list[str] = []
    cc_addresses: list[str] = []
    bcc_addresses: list[str] = []
    for address, recipient_type in cursor.fetchall():
        kind = recipient_type.lower()
        if kind == "to":
            to_addresses.append(address)
        elif kind == "cc":
            cc_addresses.append(address)
        elif kind == "bcc":
            bcc_addresses.append(address)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )


async def read_email(message_id: str) -> Email | None:
    return await asyncio.to_thread(_read_email_sync, message_id)
