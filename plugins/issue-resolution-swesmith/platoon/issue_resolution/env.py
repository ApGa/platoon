import logging

from openhands.sdk.workspace import BaseWorkspace

from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished
from platoon.issue_resolution.test_execution_reward.test_execution_reward import compute_test_execution_reward
# from platoon.issue_resolution.localization_reward.localization_reward import compute_localization_reward
# from platoon.issue_resolution.tool_error_reward.tool_json_error import compute_tool_json_error_reward
# from platoon.issue_resolution.tool_error_reward.agent_error_event import compute_agent_error_reward
logger = logging.getLogger(__name__)

def remove_binary_files_from_git():
    """
    Generate a bash command to remove binary files from git staging.
    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (file "$file" | grep -q "executable" || \\
            git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()

def extract_patch_from_environment(
    workspace: BaseWorkspace,
    repo_path: str,
):
    ex_code = workspace.execute_command(f"cd {repo_path} ; git add -A")
    git_commit = workspace.execute_command(
        f"cd {repo_path} && "
        f"git config --global user.email 'evaluation@openhands.dev' && "
        f"git config --global user.name 'OpenHands Evaluation' && "
        f'git config --global core.pager ""'
    )
    rm_binary_diff = workspace.execute_command(f"cd {repo_path} && {remove_binary_files_from_git()}")
    git_patch_result = workspace.execute_command(
        (f"cd {repo_path}; git diff --no-color --cached")
    )
    if git_patch_result.exit_code != 0:
        logger.error(f"git diff command failed with exit code {git_patch_result.exit_code} and stderr: {git_patch_result.stderr} {git_patch_result.stdout}")
    git_patch = git_patch_result.stdout
    return git_patch

class SWEBenchEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        if not is_finished(self._state):
            return 0.0, {}

        instance: dict = self._task.misc

        reward = 0.0
        info = {}

        # --- extract model patch ---
        try:
            model_patch = extract_patch_from_environment(
                workspace=self._workspace,
                repo_path=instance["repo_path"],
                # base_commit=instance.get("base_commit"),
            )
        except Exception as e:
            logger.warning("Failed to extract model patch: %s", e)
            model_patch = ""
            info["patch_extraction_error"] = "Failed to extract model patch: " + str(e)
        
        # Execute tests on Modal
        test_execution_reward, test_execution_info = await compute_test_execution_reward(model_patch, instance)

        # # Compute Localization reward
        # localization_reward, localization_reward_info = compute_localization_reward(model_patch, instance, "adityasoni17/SWE-smith-py-code-search", "train")

        # # Compute tool_json_error_reward
        # tool_json_error_reward, tool_json_error_reward_info = compute_tool_json_error_reward(self._conversation.state.events)

        # # Compute agent_error_event_reward
        # agent_error_event_reward, agent_error_event_reward_info = compute_agent_error_reward(self._conversation.state.events)

        # reward weights
        # TEST_EXECUTION_REWARD_WEIGHT = 0.70
        # LOCALIZATION_REWARD_WEIGHT = 0.20
        # TOOL_JSON_ERROR_REWARD_WEIGHT = 0.05
        # AGENT_ERROR_EVENT_REWARD_WEIGHT = 0.05

        # reward = (test_execution_reward * TEST_EXECUTION_REWARD_WEIGHT) + \
        #          (localization_reward * LOCALIZATION_REWARD_WEIGHT) + \
        #          (tool_json_error_reward * TOOL_JSON_ERROR_REWARD_WEIGHT) + \
        #          (agent_error_event_reward * AGENT_ERROR_EVENT_REWARD_WEIGHT) 

        reward = test_execution_reward
        info.update(test_execution_info)
        # info.update(localization_reward_info)
        # info.update(tool_json_error_reward_info)
        # info.update(agent_error_event_reward_info)
        return reward, info
        # if len(model_patch.strip()) == 0: # empty model patch is guaranteed to fail, so we can skip evaluation on Modal
        #     return 0.0, {"error": "Empty model patch ==> guaranteed to not resolve issues."}

        # Localization reward 
        # localization_reward_value, localization_reward_info = localization_reward(model_patch, instance, dataset="adityasoni17/SWE-smith-py-code-search", split="train")
        # # Test execution reward: run evaluation of this patch on Modal

        # event_stream = self._conversation.state.events
        


        # try:
        #     run_id = f"rl-{uuid.uuid4().hex}"
        #     with modal.enable_output():
        #         modal_fn = modal.Function.from_name("swesmith-evaluation", "run_instance_modal")
        #         res = await modal_fn.remote.aio(
        #             prediction={
        #                 KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
        #                 KEY_PREDICTION: model_patch,
        #                 KEY_MODEL: "test_model",
        #             },
        #             instance=instance,
        #             run_id=run_id,
        #             f2p_only=False,
        #             is_gold=False,
        #             timeout=5*60,
        #             verbose=False,
        #             build_image_from_scratch=False,
        #         )
        #     info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        #     try:
        #         reward = 1.0 if res.resolved else 0.0
        #         info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        #     except Exception as e:
        #         reward = 0.0
        #         info = {"model_patch": model_patch, "evaluation_logs": str(e)}
        #     return reward, info
        # except Exception as e:
        #     return 0.0, {"error": f"Failed to evaluate patch on Modal: {e}"}
        # except:
        #     return 0.0, {"error": "Failed to evaluate patch on Modal due to an unknown error."}