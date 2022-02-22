"""Module world_helpers. Specifies
various helpers so that we have more uniform interface
when using the various environments.
"""

from typing import TypeVar

Env = TypeVar('Env')
TimeStep = TypeVar('TimeStep')
Action = TypeVar('Action')


def n_states(env: Env) -> int:
    """Queries the env if it has n_states attribute
    if yes it uses it to return the number of states in the
    environment. Otherwise it uses env.observation_space.n

    Parameters
    ----------
    env: The environment to query

    Returns
    -------

    The number of states in the environment

    """

    n_states_ = getattr(env, "n_states", None)

    if n_states_ is None:
        n_states_ = env.observation_space.n
    else:
        n_states_ = env.n_states

    return n_states_


def n_actions(env: Env) -> int:
    """Queries the env if it has n_actions attribute
    if yes it uses it to return the number of actions in the
    environment. Otherwise it uses env.action_space.n

    Parameters
    ----------
    env: The environment to query

    Returns
    -------

    The number of actions in the environment

    """
    n_actions_ = getattr(env, "n_actions", None)

    if n_actions_ is None:
        n_actions_ = env.action_space.n
    else:
        n_actions_ = env.n_actions

    return n_actions_


def reset(env: Env) -> TimeStep:
    """Reset the environment

    Parameters
    ----------
    env: The environment to reset

    Returns
    -------

    An instance to TimeStep

    """
    state, reward, done, info = env.reset()
    time_step = TimeStep(state=state, reward=reward, done=done, info=info)
    return time_step


def step(env: Env, action: Action) -> TimeStep:
    """Step in the environment using the given action

    Parameters
    ----------
    env: The environment to step
    action: The action to execute

    Returns
    -------
    An instance to TimeStep

    """
    state, reward, done, info = env.step(action)
    time_step = TimeStep(state=state, reward=reward, done=done, info=info)
    return time_step


