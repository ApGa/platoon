from pathlib import Path
from typing import List, Tuple
from platoon.utils.openhands_utils import is_finished
from platoon.episode.context import current_trajectory_collection, current_trajectory, finish_message, error_message

from openhands.sdk import get_logger
from platoon.envs.base import Task
from openhands.sdk.agent import AgentBase
from openhands.sdk.workspace import BaseWorkspace
from openhands.sdk.conversation import Conversation, BaseConversation, get_agent_final_response
#TODO: check below imports
from platoon.openhands.env import OpenHandsEnv
from platoon.openhands.types import OpenHandsObservation, OpenHandsAction, OpenHandsTrajectoryStep
import threading
import asyncio
from platoon.utils.openhands_utils import is_finished
from openhands.sdk.event import ActionEvent
from platoon.codescout.localization_finish import LocalizationFinishAction

def get_structured_locations(events):
    """Extract structured locations from LocalizationFinishAction in events.
    Args:
        events: List of conversation events to search through.
    Returns:
        List of location dicts with 'file', 'class', 'function' keys, or None if not found.
    """
    # Find the last LocalizationFinishAction
    cnt = [1 for event in events if isinstance(event, ActionEvent) and event.source == "agent" and isinstance(event.action, LocalizationFinishAction)]
    cnt = sum(cnt)
    if cnt != 1: # the localization finish tool must be called exactly once.
        return None
    for event in reversed(events):
        if (
            isinstance(event, ActionEvent)
            and event.source == "agent"
            and isinstance(event.action, LocalizationFinishAction)
        ):
            # Extract structured locations from the action
            locations = []
            for loc in event.action.locations:
                locations.append({
                    "file": loc.file,
                    "class_name": loc.class_name,
                    "function_name": loc.function_name,
                })
            return locations
    return None

def parse_structured_outputs(structured_locations: List[dict]) -> Tuple[List[str], List[str], List[str]]:
    """
    Process structured location outputs and extract files, modules, and entities.

    Args:
        structured_locations: List of dicts with 'file', 'class_name', 'function_name' keys
        Returns:
            Tuple of (all_found_files, all_found_modules, all_found_entities) where each is a list of strs
    
    Example structured input format:
        [
            {'file': 'path/to/file1.py', 'class_name': 'MyClass', 'function_name': 'my_method'},
            {'file': 'path/to/file2.py', 'class_name': None, 'function_name': 'standalone_function'},
            {'file': 'path/to/file1.py', 'class_name': None, 'function_name': 'global_function'},
            {'file': 'path/to/file2.py', 'class_name': 'AnotherClass', 'function_name': None},
            {'file': 'path/to/file3.py', 'class_name': None, 'function_name': None}
        ]
    
    Example output:
        [['path/to/file1.py', 'path/to/file2.py', 'path/to/file3.py'], ['path/to/file1.py:MyClass', 'path/to/file2.py:AnotherClass', 'path/to/file1.py:global_function', 'path/to/file2.py:standalone_function'], ['path/to/file1.py:MyClass.my_method', 'path/to/file2.py:standalone_function', 'path/to/file1.py:global_function', 'path/to/file2.py:AnotherClass']]
    """

    # Strict sanity check: if there are duplicates in the output, return an empty output so that it is penalized with 0 reward?
    all_found_files = []
    all_found_modules = []
    all_found_entities = []

    found_empty_filename = False
    # found_duplictes = False

    for location in structured_locations:
        file_path = location.get("file", None)
        class_name = location.get("class_name", None)
        function_name = location.get("function_name", None)

        #NOTE: Ideally the case of file_path being None should raise an error from the agent-sdk but adding here for safety
        if file_path is None or file_path.strip() == "":
            found_empty_filename = True
            break

        all_found_files.append(file_path)

        module = None
        if class_name:
            module = f"{file_path}:{class_name}"
        elif function_name:
            module = f"{file_path}:{function_name}"
        
        if module:
            all_found_modules.append(module)

        entity = None
        if class_name and function_name:
            entity = f"{file_path}:{class_name}.{function_name}"
        elif function_name:
            entity = f"{file_path}:{function_name}"

        if entity:
            all_found_entities.append(entity)
    if found_empty_filename:
        return [], [], []
    all_found_files = list(set(all_found_files))
    all_found_modules = list(set(all_found_modules))
    all_found_entities = list(set(all_found_entities))
    return all_found_files, all_found_modules, all_found_entities

def compute_file_f1_score(predicted_files, true_files, beta=1.0):
    pred, true = set(predicted_files), set(true_files)
    if not true:
        return 0.0 # return 0 reward if ground truth is empty
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

def multilevel_localization_f1_reward(
    instance: dict,
    structured_locations: list[dict] | None = None,
    file_level_weight: float=1.0,
    module_level_weight: float=1.0,
    entity_level_weight: float=1.0,
):

    if structured_locations is None:
        return 0, {
        "multilevel_localization_f1_reward": 0,
        "file_reward": 0,
        "module_reward": 0,
        "entity_reward": 0,
    }

    gt_files = []
    gt_modules = []
    gt_entities = []
    reward = 0

    for change in instance.get("file_changes", []):
        if "file" in change:
            gt_files.append(change["file"])
        if "changes" in change:
            edited_modules = change["changes"].get("edited_modules", [])
            edited_modules = [] if edited_modules is None else edited_modules
            for module in edited_modules:
                gt_modules.append(module)

            edited_entities = change["changes"].get("edited_entities", [])
            edited_entities = [] if edited_entities is None else edited_entities
            for entity in edited_entities:
                gt_entities.append(entity)
    gt_files = set(gt_files)
    gt_modules = set(gt_modules)
    gt_entities = set(gt_entities)

    if structured_locations is not None:
        predicted_files, predicted_modules, predicted_entities = parse_structured_outputs(structured_locations)
    else:
        predicted_files, predicted_modules, predicted_entities = get_simple_results_from_raw_outputs(final_message)

    file_f1_score = compute_file_f1_score(predicted_files, gt_files)
    module_f1_score = compute_file_f1_score(predicted_modules, gt_modules)
    entity_f1_score = compute_file_f1_score(predicted_entities, gt_entities)

    reward = (
        file_f1_score * file_level_weight
    + module_f1_score * module_level_weight
    + entity_f1_score * entity_level_weight
    )

    return reward, {
        "multilevel_localization_f1_reward": reward,
        "file_reward": file_f1_score,
        "module_reward": module_f1_score,
        "entity_reward": entity_f1_score,
    }

class CodeScoutEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        if not is_finished(await self.observe()):
            return 0, {}
        
        structured_locations = get_structured_locations(self._conversation.state.events)

        if structured_locations is None:
            return 0, {}
        
        instance: dict = self.task.misc
        reward, metadata = multilevel_localization_f1_reward(instance, structured_locations)
        return reward, metadata