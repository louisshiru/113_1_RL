from .task import Task
import numpy as np

class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0, collision_reward: float = 0.0,
                 frame_reward: float = 0.0, progress_reward: float = 100.0, n_min_rays_termination=1080):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        self._last_stored_progress = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = abs(progress - self._last_stored_progress)
        if delta > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward
        if self._check_collision(agent_state):
            reward += self._collision_reward
        reward += delta * self._progress_reward
        self._last_stored_progress = progress
        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None


class MaximizeProgressMaskObstacleTask(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
                 collision_reward=0, frame_reward=0, progress_reward=100):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)

    def reward(self, agent_id, state, action) -> float:
        progress_reward = super().reward(agent_id, state, action)
        distance_to_obstacle = state[agent_id]['obstacle']
        if distance_to_obstacle < .3:  # max distance = 1, meaning perfectly centered in the widest point of the track
            return 0.0
        else:
            return progress_reward


class MaximizeProgressRegularizeAction(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
                 collision_reward=0, frame_reward=0, progress_reward=100, action_reg=0.25):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)
        self._action_reg = action_reg
        self._last_action = None

    def reset(self):
        super(MaximizeProgressRegularizeAction, self).reset()
        self._last_action = None

    def reward(self, agent_id, state, action) -> float:
        """ Progress-based with action regularization: penalize sharp change in control"""
        reward = super().reward(agent_id, state, action)
        action = np.array(list(action.values()))
        if self._last_action is not None:
            reward -= self._action_reg * np.linalg.norm(action - self._last_action)
        self._last_action = action
        return reward


class RankDiscountedMaximizeProgressTask(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.001,
                 collision_reward=-100, frame_reward=-0.1, progress_reward=1):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)

    def reward(self, agent_id, state, action) -> float:
        rank = state[agent_id]['rank']
        reward = super().reward(agent_id, state, action)
        reward = reward / float(rank)
        return reward


class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0, collision_reward: float = 0.0,
                 frame_reward: float = 0.0, progress_reward: float = 100.0, n_min_rays_termination=1080):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        self._last_stored_progress = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = abs(progress - self._last_stored_progress)
        if delta > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward
        if self._check_collision(agent_state):
            reward += self._collision_reward
        reward += delta * self._progress_reward
        self._last_stored_progress = progress
        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None


class MaximizeProgressTaskCollisionInfluenceTimeLimit(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0,
                 collision_reward: float = 0.0,
                 frame_reward: float = 0.0,
                 progress_reward: float = 100.0,
                 n_min_rays_termination=1080,
                 collision_penalty_time_reduce=40.):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        self._last_stored_progress = None
        self._last_actions         = None
        self._last_state           = None
        self._last_reward          = 0
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward
        self.n_collision = 0
        self.collision_penalty_time_reduce = collision_penalty_time_reduce
        self.idle_frames = 0
        self.expert_ratio = 0.01

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
                
        """ Origin
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = abs(progress - self._last_stored_progress)
        if delta > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward
        if self._check_collision(agent_state):
            self.n_collision += 1
            reward += self._collision_reward
        reward += delta * self._progress_reward
        self._last_stored_progress = progress
        """     
        
        # """ modified
        if self._last_state is None:
            self._last_state = agent_state
        
        if self._last_actions is None:
            self._last_actions = action
            
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
                        
        delta = progress - self._last_stored_progress
        if abs(delta) > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward # 0
        if self._check_collision(agent_state):
            self.n_collision += 1
            reward += self._collision_reward
        reward += delta * self._progress_reward * 0
        
        reward = 0.
        # if agent_state['wall_collision'] == True:
        #     reward -= 100
            
        # if agent_state['progress'] >= (1 - 1e-3):
        #     reward += 100
        # if agent_state['checkpoint'] > self._last_state['checkpoint']:
        #     reward += 15
                
        # reward += 1 * action['motor_competition']
        # reward -= 0.1 * (abs(action['motor_competition'] - self._last_actions['motor_competition']) + abs(action['steering_competition'] - self._last_actions['steering_competition']))
        # if agent_state['progress'] > self._last_state['progress']:
        #     reward += 1000 * (agent_state['progress'] - self._last_state['progress'])
        # elif agent_state['progress'] <= self._last_state['progress']:
        #     reward -= 0.1
        
        if agent_state['velocity'][0] < 0.1:
            self.idle_frames += 1
            reward -= 2
        else:
            self.idle_frames = 0 
        
        reward += 250 * delta
        
        if self._check_collision(agent_state) == True:
            reward -= 50
        
        self._last_stored_progress = progress
        self._last_actions = action.copy()
        self._last_state = agent_state.copy()
        self._last_reward = reward
                
        return reward

    def _check_idle(self):
        time_done = False
        if self.idle_frames >= 30:
            self.idle_frames = 0
            time_done = True
        return time_done

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        # Collision reduce the time limit
        total_penalty = sum(agent_state['collision_penalties'])
        lap_done = agent_state['lap'] > self._laps
        time_done = (self._time_limit - total_penalty) < agent_state['time']
        
        time_done = time_done or self._check_idle()
            
        return lap_done or time_done

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        
        return collision

    def reset(self):
        self._last_stored_progress = None
        self.n_collision = 0
        