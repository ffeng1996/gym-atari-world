import logging
from gym.envs.registration import register, registry, make, spec

logger = logging.getLogger(__name__)

register(
    id='atariWorld-v0000',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 1},
)

register(
    id='atariWorld-v1000',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 255},
)

register(
    id='atariWorld-v0010',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'orientation': False},
)

register(
    id='atariWorld-v0001',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'noise': True},
)

register(
    id='atariWorld-v0100',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'size': 10},
)


