import random
from typing import Dict, Any, Tuple, Optional, SupportsFloat, Union
import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from racecar_gym.bullet.positioning import RecoverPositioningStrategy
from racecar_gym.envs.scenarios import SingleAgentScenario


# noinspection PyProtectedMember
class SingleAgentRaceEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenario: str, render_mode: str = 'human', render_options: Optional[Dict[str, Any]] = None,
                 reset_when_collision: bool = True,
                 collision_penalty_weight: float = 10.):
        self.scenario_name = scenario
        scenario = SingleAgentScenario.from_spec(scenario, rendering=render_mode == 'human')
        self._scenario = scenario
        self._initialized = False
        self._render_mode = render_mode
        self._render_options = render_options or {}
        self.action_space = scenario.agent.action_space
        self.reset_when_collision = reset_when_collision

        self.recover_strategy = RecoverPositioningStrategy(progress_map=self._scenario.world._maps['progress'],
                                                           obstacle_map=self._scenario.world._maps['obstacle'],
                                                           alternate_direction=False)
        self.collision_penalties = []
        self.collision_penalty_weight = collision_penalty_weight

    @property
    def observation_space(self):
        space = self._scenario.agent.observation_space
        # space.spaces['time'] = gymnasium.spaces.Box(low=0, high=1, shape=(1,)),
        return space

    @property
    def scenario(self):
        return self._scenario

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert self._initialized, 'Reset before calling step'
        observation, info = self._scenario.agent.step(action=action)

        self._scenario.world.update()

        cur_state = self._scenario.world.state()[self.scenario.agent._id]
        if self.reset_when_collision and self._scenario.agent.task._check_collision(cur_state):
            if 'austria' in self.scenario_name:
                cur_progress = cur_state['progress']
                collision_penalty = 30 + np.sum(cur_state['velocity'] ** 2) * self.collision_penalty_weight
                self.collision_penalties.append(collision_penalty)
                recover_pose = self.recover_strategy.get_recover_pose(cur_progress)
                self._scenario.agent._vehicle.reset(pose=recover_pose)
            else:
                raise ValueError('Recover are only supported for austria scenario')

        state = self._scenario.world.state()
        info = state[self._scenario.agent.id]
        if hasattr(self._scenario.agent.task, 'n_collision'):
            info['n_collision'] = self._scenario.agent.task.n_collision
        # observation['time'] = np.array([state[self._scenario.agent.id]['time']], dtype=np.float32)
        state[self._scenario.agent.id]['collision_penalties'] = np.array(self.collision_penalties)
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        info['reward'] = reward
        return observation, reward, done, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.collision_penalties = []
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        if options is not None and 'mode' in options:
            mode = options['mode']
        else:
            mode = 'grid'
        pos = self._scenario.world.get_starting_position(self._scenario.agent, mode)
        
        # lists = []
        # for i in range(200):
        #     pos = self._scenario.world.get_starting_position(self._scenario.agent, 'random') # uncomment for normal
        #     obs = self._scenario.agent.reset(pos)
        #     self._scenario.world.update()
        #     state = self._scenario.world.state()
        #     state[self._scenario.agent.id]['collision_penalties'] = np.array(self.collision_penalties)
        #     info = state[self._scenario.agent.id]
        #     if 0.29 <= info['progress'] and info['progress'] <= 0.39:
        #         lists.append((pos, info['progress']))
    
        pos_list = [
                ((15.650000000000006, -17.299999999999997, 0.05), (0, 0, -1.9624962662323215)),  
                # ((13.950000000000003, -13.549999999999997, 0.05), (0, 0, -1.197808604994834)),   
                # ((13.950000000000003, -13.549999999999997, 0.05), (0, 0, -1.197808604994834)),   
                # ((16.85000000000001, -19.4, 0.05), (0, 0, 2.894495349565077)),                   
                ((15.650000000000006, -20.0, 0.05), (0, 0, 2.7149651604629175)),                 
                # ((16.799999999999997, -19.25, 0.05), (0, 0, 3.012681881905801)),                 
                ((15.450000000000003, -16.549999999999997, 0.05), (0, 0, -1.0943289073211915)),  
                ((15.549999999999997, -17.4, 0.05), (0, 0, -1.6666554739049098)),                
                # ((15.5, -17.299999999999997, 0.05), (0, 0, -1.8878520800040426)),                
                # ((12.850000000000001, -19.599999999999998, 0.05), (0, 0, 2.6399885994006733)),   
                # ((14.950000000000003, -16.4, 0.05), (0, 0, -1.7792615451748832)),                
                # ((14.5, -15.449999999999996, 0.05), (0, 0, -1.4077010345846546)),                
                # ((11.5, -18.799999999999997, 0.05), (0, 0, 2.758849143744218)),                  
                # ((16.049999999999997, -17.75, 0.05), (0, 0, -2.508843818587609)),               
                # ((12.100000000000001, -19.349999999999998, 0.05), (0, 0, 2.8879185574511506)),   
                # ((11.800000000000004, -18.65, 0.05), (0, 0, 2.963499715358596)),                 
                # ((15.549999999999997, -17.25, 0.05), (0, 0, -2.2005996622153132)),               
                # ((14.450000000000003, -19.45, 0.05), (0, 0, 3.07502448981397)),                  
                # ((16.400000000000006, -18.599999999999998, 0.05), (0, 0, -1.6124389058934885)),  
                # ((15.150000000000006, -16.699999999999996, 0.05), (0, 0, -1.0496581833107796)),  
                # ((15.299999999999997, -19.7, 0.05), (0, 0, 2.7435783379609413)),                 
                # ((15.700000000000003, -17.949999999999996, 0.05), (0, 0, -2.677945044588986)),   
                # ((14.799999999999997, -15.399999999999999, 0.05), (0, 0, -0.9750117791718376)),  
                # ((16.0, -19.049999999999997, 0.05), (0, 0, 2.938547436336328)),                  
                # ((16.0, -19.549999999999997, 0.05), (0, 0, 3.0471377181947465)),                 
                # ((12.950000000000003, -19.549999999999997, 0.05), (0, 0, 3.0443394553382275)),   
                # ((15.049999999999997, -15.699999999999996, 0.05), (0, 0, -1.2178059389679827)),  
                # ((15.850000000000009, -20.049999999999997, 0.05), (0, 0, 3.0060649396042933)),   
                # ((13.25, -19.25, 0.05), (0, 0, 2.5231646592172723)),                             
                # ((14.5, -19.549999999999997, 0.05), (0, 0, 2.6167968819396856)),                 
                # ((14.350000000000009, -19.849999999999998, 0.05), (0, 0, 2.7856349154218085)),   
                # ((12.700000000000003, -19.45, 0.05), (0, 0, 2.5171823039631525)),                
                # ((16.400000000000006, -18.599999999999998, 0.05), (0, 0, -1.2560386654035869)),  
                # ((14.75, -16.1, 0.05), (0, 0, -0.8574382287842758)),                             
                # ((15.299999999999997, -20.0, 0.05), (0, 0, 2.7209293044567575)), 
                # ((15.650000000000006, -17.299999999999997, 0.05), (0, 0, -1.9624962662323215))                
        ]
        # pos = random.choice(pos_list)
    
        obs = self._scenario.agent.reset(pos)
        self._scenario.world.update()
        state = self._scenario.world.state()
        state[self._scenario.agent.id]['collision_penalties'] = np.array(self.collision_penalties)
        info = state[self._scenario.agent.id]
        # breakpoint()
        if hasattr(self._scenario.agent.task, 'n_collision'):
            info['n_collision'] = self._scenario.agent.task.n_collision
        # obs['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        info['reward'] = 0.
        return obs, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        if self._render_mode == 'human':
            return None
        else:
            mode = self._render_mode.replace('rgb_array_', '')
            return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **self._render_options)

    def force_render(self, render_mode: str, **kwargs):
        mode = render_mode.replace('rgb_array_', '')
        return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **kwargs)
