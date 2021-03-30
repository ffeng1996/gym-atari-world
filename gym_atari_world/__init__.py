import logging
from gym.envs.registration import register, registry, make, spec

logger = logging.getLogger(__name__)

# Change in color: v0*

register(
    id='atariWorld-v00',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 1},
)

register(
    id='atariWorld-v01',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 50},
)
register(
    id='atariWorld-v02',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 255},
)


register(
    id='atariWorld-v10',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 1, 'orientation': False},
)

register(
    id='atariWorld-v101',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 1, 'orientation': False, 'size': 3},
)


