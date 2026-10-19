"""TextCraft: crafting environments (ORS) for tool-using and recursive agents, with platoon's named environments.

``env`` serves TextCraft and TextCraft-Synth tasks and imports neither platoon nor IPython, so any ORS client can
run it.  ``prompts`` holds the CodeAct prompts of the linear, recursive and depth-aware agents, ``entries`` the named
environments ``textcraft@1`` and ``textcraft-synth@1``, and ``tasks`` and ``synth_tasks`` the task generators:
those need platoon (the ``platoon`` extra).  ``harbor`` exports TextCraft-Synth as a Harbor dataset.  This module
imports none of them.
"""
