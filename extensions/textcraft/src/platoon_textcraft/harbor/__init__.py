"""TextCraft as a Harbor dataset: the in-container game (``game``) and the exporter (``export``).

``python -m platoon_textcraft.harbor export --variant synth --out DIR --version 1.0`` writes TextCraft-Synth
(``textcraft-synth@1``) as Harbor task directories with a registry, a task index, split lists and a dataset card.
``game`` is the standalone ``textcraft`` command and verifier that every task carries (Python's standard library
only, no platoon import).
"""
