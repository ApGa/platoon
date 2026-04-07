from __future__ import annotations

from pydantic import BaseModel, Field


class SyntheticQuery(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: list[str]
    how_realistic: float
    inbox_address: str
    query_date: str


class Email(BaseModel):
    message_id: str
    date: str
    subject: str | None = None
    from_address: str | None = None
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    bcc_addresses: list[str] = Field(default_factory=list)
    body: str | None = None
    file_name: str | None = None
