from algorithms.policy import Policy


class PolicySampler(Policy):
    pass


class DummyPolicySampler(PolicySampler):

    def __init__(self) -> None:
        super(DummyPolicySampler, self).__init__()

    def __call__(self, state):
        return None
