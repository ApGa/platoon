from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import datetime

from datasets import Dataset, Features, Sequence, Value, load_dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")
DB_PATH_ENV_VAR = "PLATOON_EMAIL_SEARCH_DB_PATH"
DEFAULT_REPO_ID = "corbt/enron-emails"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 message_id TEXT UNIQUE,
 subject TEXT,
 from_address TEXT,
 date TEXT,
 body TEXT,
 file_name TEXT
);

CREATE TABLE recipients (
 email_id INTEGER,
 recipient_address TEXT,
 recipient_type TEXT,
 FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

CREATE VIRTUAL TABLE emails_fts USING fts5(
 subject,
 body,
 content='emails',
 content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
 INSERT INTO emails_fts (rowid, subject, body)
 VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
 DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
 UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""


def resolve_db_path(db_path: str | None = None) -> str:
    """Resolve the sqlite path, allowing an override outside the repo."""
    return os.path.abspath(db_path or os.environ.get(DB_PATH_ENV_VAR, DEFAULT_DB_PATH))


def download_dataset(repo_id: str = DEFAULT_REPO_ID) -> Dataset:
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset_obj = load_dataset(repo_id, features=expected_features, split="train")
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset_obj)}")
    return dataset_obj


def create_database(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SQL_CREATE_TABLES)
        conn.commit()
    finally:
        conn.close()


def populate_database(db_path: str, dataset: Dataset) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA journal_mode = MEMORY;")
        conn.execute("BEGIN TRANSACTION;")

        processed_emails: set[tuple[str | None, str, str | None]] = set()

        for email_data in tqdm(dataset, desc="Inserting emails"):
            assert isinstance(email_data, dict)
            message_id = str(email_data["message_id"])
            subject = email_data["subject"]
            from_address = email_data["from"]
            date_obj: datetime = email_data["date"]
            body = str(email_data["body"])
            file_name = email_data["file_name"]

            to_list = [str(addr) for addr in email_data["to"] if addr]
            cc_list = [str(addr) for addr in email_data["cc"] if addr]
            bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

            if len(body) > 5000:
                continue

            total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
            if total_recipients > 30:
                continue

            email_key = (subject, body, from_address)
            if email_key in processed_emails:
                continue
            processed_emails.add(email_key)

            cursor.execute(
                """
                INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    subject,
                    from_address,
                    date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                    body,
                    file_name,
                ),
            )
            email_pk_id = cursor.lastrowid

            recipient_rows: list[tuple[int, str, str]] = []
            recipient_rows.extend((email_pk_id, addr, "to") for addr in to_list)
            recipient_rows.extend((email_pk_id, addr, "cc") for addr in cc_list)
            recipient_rows.extend((email_pk_id, addr, "bcc") for addr in bcc_list)

            if recipient_rows:
                cursor.executemany(
                    """
                    INSERT INTO recipients (email_id, recipient_address, recipient_type)
                    VALUES (?, ?, ?)
                    """,
                    recipient_rows,
                )

        conn.commit()
    finally:
        conn.close()


def create_indexes_and_triggers(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SQL_CREATE_INDEXES_TRIGGERS)
        conn.commit()
    finally:
        conn.close()


def generate_database(overwrite: bool = False, db_path: str | None = None) -> str:
    resolved_db_path = resolve_db_path(db_path)
    db_dir = os.path.dirname(resolved_db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    if overwrite and os.path.exists(resolved_db_path):
        os.remove(resolved_db_path)
    elif not overwrite and os.path.exists(resolved_db_path):
        logging.info("Email database already exists at %s", resolved_db_path)
        return resolved_db_path

    dataset = download_dataset(DEFAULT_REPO_ID)
    create_database(resolved_db_path)
    populate_database(resolved_db_path, dataset)
    create_indexes_and_triggers(resolved_db_path)
    return resolved_db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the local email-search sqlite database.")
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            f"Custom sqlite output path. Defaults to `{DB_PATH_ENV_VAR}` when set, "
            "otherwise the repo-local data directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing database at the target path.",
    )
    args = parser.parse_args()
    generate_database(overwrite=args.overwrite, db_path=args.db_path)


if __name__ == "__main__":
    main()
