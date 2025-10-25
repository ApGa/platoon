from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.envs.base import Task
from platoon.agents.actions.common import finish
from platoon.envs.countdown.eval import compute_score
from platoon.episode.context import finish_message

class CountDownEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task, IPythonCodeExecutor(task, actions=(finish,)))
        
    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0., {}
        if self._state.finished:
            score = compute_score(finish_message.get(), self._task.misc, format_score=0.0)
        return score, reward_misc
