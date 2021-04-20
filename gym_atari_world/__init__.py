import logging
from gym.envs.registration import register, registry, make, spec

logger = logging.getLogger(__name__)

'''
Changing factors: v_{color}_{size}_{orientation}_{noise}
0: No change, set to default
1: Change
'''
register(
    id='atariWorld-v1000',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 0.5},
)

register(
    id='atariWorld-v0100',
    entry_point='gym_atari_world.envs:AtariEnv',
    kwargs={'color': 2},
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



