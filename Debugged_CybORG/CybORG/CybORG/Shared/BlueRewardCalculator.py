from collections import namedtuple

from CybORG.Shared import Scenario
from CybORG.Shared.RedRewardCalculator import (DistruptRewardCalculator, PwnRewardCalculator,
                                               RansomwareRewardCalculator, CryptominerRewardCalculator)
from CybORG.Shared.RewardCalculator import RewardCalculator

HostReward = namedtuple('HostReward','confidentiality availability')


class ConfidentialityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on compromise of hosts/data
    def __init__(self, agent_name: str, scenario: Scenario):
        self.scenario = scenario
        self.adversary = scenario.get_agent_info(agent_name).adversary
        super(ConfidentialityRewardCalculator, self).__init__(agent_name)
        self.infiltrate_rc = PwnRewardCalculator(self.adversary, scenario)
        self.compromised_hosts = {}

    def reset(self):
        self.infiltrate_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.compromised_hosts = {}
        reward = -self.infiltrate_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_compromised_hosts()
        return reward

    def _calculate_compromised_hosts(self):
        for host, value in self.infiltrate_rc.compromised_hosts.items():
            self.compromised_hosts[host] = -1 * value


class AvailabilityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on reduction in availability
    def __init__(self, agent_name: str, scenario: Scenario):
        super(AvailabilityRewardCalculator, self).__init__(agent_name)
        self.adversary = scenario.get_agent_info(agent_name).adversary
        self.disrupt_rc = DistruptRewardCalculator(self.adversary, scenario)
        self.impacted_hosts = {}

    def reset(self):
        self.disrupt_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.impacted_hosts = {}
        reward = -self.disrupt_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_impacted_hosts()
        return reward

    def _calculate_impacted_hosts(self):
        for host, value in self.disrupt_rc.impacted_hosts.items():
            self.impacted_hosts[host] = -1 * value


class HybridAvailabilityConfidentialityRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality reward calculator
    def __init__(self, agent_name: str, scenario: Scenario):
        super(HybridAvailabilityConfidentialityRewardCalculator, self).__init__(agent_name)
        self.availability_calculator = AvailabilityRewardCalculator(agent_name, scenario)
        self.confidentiality_calculator = ConfidentialityRewardCalculator(agent_name, scenario)

    def reset(self):
        self.availability_calculator.reset()
        self.confidentiality_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = self.availability_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.confidentiality_calculator.calculate_reward(current_state, action, agent_observations, done)
        self._compute_host_scores(current_state.keys())
        return reward

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.confidentiality_calculator.compromised_hosts
        impacted_hosts = self.availability_calculator.impacted_hosts
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            reward_state = HostReward(compromised,impacted)  
                                    # confidentiality, availability
            self.host_scores[host] = reward_state


class RansomwareDefenseCalculator(RewardCalculator):
    # Specialized calculator for computing ransomware defensive rewards
    def __init__(self, agent_name: str, scenario: Scenario):
        super(RansomwareDefenseCalculator, self).__init__(agent_name)
        self.default_calc = HybridAvailabilityConfidentialityRewardCalculator(agent_name, scenario)
        self.ransomware_calc = RansomwareRewardCalculator(agent_name, scenario)
        self.compromised_hosts = dict()

    def reset(self):
        self.default_calc.reset()
        self.ransomware_calc.reset()
        self.compromised_hosts = dict()

    def _compute_host_scores(self, hostnames):
        self.default_calc._compute_host_scores(hostnames)
        self.host_scores = self.default_calc.host_scores
        self.compromised_hosts = self.ransomware_calc.compromised_hosts
        for host in hostnames:
            if host == 'success':
                continue
            if host in self.compromised_hosts.keys():
                compromised = compromised_hosts[host] if host in compromised_hosts else 0
                self.host_scores[host].confidentiality = self.compromised_hosts[host].confidentiality - compromised
                self.host_scores[host].availability = self.compromised_hosts[host].availability - compromised

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = (self.default_calc.calculate_reward(current_state, action, agent_observations, done) -
                  self.calculate_ransomware_reward(current_state, action, agent_observations, done))
        self._compute_host_scores(current_state.keys())
        return reward

    def calculate_ransomware_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = -self.ransomware_calc.calculate_reward(current_state, action, agent_observations, done)
        return reward


class CryptominerDefenseCalculator(RewardCalculator):
    # Specialized calculator for computing cryptominer defensive rewards
    def __init__(self, agent_name: str, scenario: Scenario):
        super(CryptominerDefenseCalculator, self).__init__(agent_name)
        self.default_calc = HybridAvailabilityConfidentialityRewardCalculator(agent_name, scenario)
        self.cryptominer_calc = CryptominerRewardCalculator(agent_name, scenario)
        self.compromised_hosts = dict()

    def reset(self):
        self.default_calc.reset()
        self.cryptominer_calc.reset()
        self.compromised_hosts = dict()

    def _compute_host_scores(self, hostnames):
        self.default_calc._compute_host_scores(hostnames)
        self.host_scores = self.default_calc.host_scores
        self.compromised_hosts = self.cryptominer_calc.compromised_hosts
        for host in hostnames:
            if host == 'success':
                continue
            if host in self.compromised_hosts.keys():
                compromised = compromised_hosts[host] if host in compromised_hosts else 0
                self.host_scores[host].confidentiality = self.compromised_hosts[host].confidentiality - compromised

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = (self.default_calc.calculate_reward(current_state, action, agent_observations, done) -
                  self.calculate_cryptominer_reward(current_state, action, agent_observations, done))
        self._compute_host_scores(current_state.keys())
        return reward

    def calculate_cryptominer_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = -self.cryptominer_calc.calculate_reward(current_state, action, agent_observations, done)
        return reward