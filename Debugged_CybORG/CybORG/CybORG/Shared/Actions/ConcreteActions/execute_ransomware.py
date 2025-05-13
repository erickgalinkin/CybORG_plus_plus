from CybORG.Shared import Observation
from CybORG.Shared.Actions.ConcreteActions.ConcreteAction import ConcreteAction
from CybORG.Simulator.State import State

class ExecuteRansomware(ConcreteAction):
    def __init__(self, session: int, agent: str):
        super().__init__(session, agent)

    def sim_execute(self, state: State) -> Observation:
        obs = Observation()
        obs.set_success(True)
        obs.add_key_value("done", True)
        return obs