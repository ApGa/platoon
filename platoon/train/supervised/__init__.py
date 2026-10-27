"""Supervised datasets: the chat examples an sft run fine-tunes its policy on (``trainer: sft``,
:mod:`platoon.train.trainer.sft`), in LLaMA-Factory's dataset conventions.

``options`` is what ``trainer_options`` sets under ``trainer: sft`` (``SupervisedOptions``); ``registry`` reads a
dataset directory's ``dataset_info.json`` (``DatasetEntry``: where a dataset's rows are, and how to read them);
``formats`` turns a row into a conversation (``Conversation``), in the ``alpaca`` or the ``sharegpt`` format;
``loading`` loads the datasets the options name into one list of conversations; ``rendering`` renders a conversation
with the policy's renderer into the datum it trains on, whose weights mark the turns that train; and ``cursor`` walks
the conversations epoch by epoch in a seeded order, as a participant of every full checkpoint, so that a resume trains
each conversation once per epoch.  Only ``rendering`` imports Tinker.
"""
