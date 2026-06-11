"""
Swarmlift environment
"""
import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# Physical Constants
ARENA_SIZE = 30.0           # Arena spans [-15, 15] in x and y
MAX_THRUST = 1.0            # Maximum thrust magnitude per drone per axis
PAYLOAD_MASS = 3.0          # Total payload mass
LINEAR_DAMPING =  0.8       # Damping on linear velocity (simulates drag)
ANGULAR_DAMPING = 0.3       # Damping on angular velocity
DT = 0.05                   # Simulation timestep (seconds)
MAX_STEPS = 800            # Episode length cap
GOAL_RADIUS = 1.0           # Success threshold: centroid within this of goal
GOAL_THETA = 0.0            # target orientation (radians)
GOAL_THETA_TOL = 0.3        # how level it must arrive

# Total payload mass split equally among the three drones.
PAYLOAD_MASS = 3.0
DRONE_MASS = PAYLOAD_MASS / 3.0

# Equilateral template: noise small -> easy, large -> scalene/hard
_EQ = np.array([[0.0, 1.7],
                [-1.5, -0.85],
                [ 1.5, -0.85]],
               dtype=np.float32) 

def sample_triangle(rng, noise=0.4, min_area=0.5):
    while True:
        triangle = (_EQ + rng.normal(0, noise, size=(3,2)).astype(np.float32))
        e1 = triangle[1] - triangle[0]
        e2 = triangle[2] - triangle[0]
        area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        if area >= min_area:    
            return triangle
        
# Fixed start and goal positions (payload centroid)
START_POS = np.array([-12.0, 0.0], dtype=np.float32)
END_POS = np.array([12.0, 0.0], dtype=np.float32)

# Creates a matrix that rotates by angle theta
def rot_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

class SwarmLiftEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "swarmlift_v0"}
    
    def __init__(self, render_mode=None, noise = 0.4):
        self.possible_agents = ["drone_0", "drone_1", "drone_2"]   
        self.agents = []                                            
        self.render_mode = render_mode

        self.noise = noise
        self.np_random = None
        self.triangle_local = None
        self.payload_inertia = None

        # Physics state (initialized in reset)
        self.payload_pos = None
        self.payload_vel = None
        self.payload_theta = None
        self.payload_omega = None
        self.timestep = None

        # Pygame state
        self._screen = None
        self._clock = None

    @functools.cache
    def observation_space(self, agent):
        # Returns a 17-dim observation vector. See _get_obs() for 
        # the exact element ordering and physical interpretation.
        return spaces.Box(low=-np.inf, high=np.inf, shape=(17,), 
                          dtype=np.float32)
    
    @functools.cache
    def action_space(self, agent):
        return spaces.Box(low=-MAX_THRUST, high=MAX_THRUST, shape=(2,), 
                          dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng()

        if options and "triangle" in options:
            triangle = np.asarray(options["triangle"], dtype=np.float32) # fixed shape for evaluation
        else:
            triangle = sample_triangle(self.np_random, noise=self.noise) # sampled for training
        self.triangle_local = triangle - triangle.mean(axis=0)
        self.payload_inertia = float(
            np.sum(DRONE_MASS * np.sum(self.triangle_local ** 2, axis=1))
        )
        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.payload_pos = START_POS.copy()
        self.payload_vel = np.zeros(2, dtype=np.float32)
        self.payload_theta = 0.0
        self.payload_omega = 0.0

        observations = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}

        return observations, infos
    
    def step(self, actions):
        # Computes where each drone is after rotation
        R = rot_matrix(self.payload_theta)
        world_offsets = self.triangle_local @ np.transpose(R)           # Shape (3,2)

        # Sum forces and torques from all drones. Net force determines linear
        # acceleration. Net torque determines angular acceleration.
        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0

        for i, agent in enumerate(self.agents):
            thrust = np.clip(actions[agent], -MAX_THRUST, 
                             MAX_THRUST).astype(np.float32)       
            total_force += thrust
            
            # Torque = lever (r) * force(F). In 2D, cross product simplifies to
            # r_x * F_y - r_y * F_x. Positive torque = counterclockwise rotation.
            r = world_offsets[i]
            total_torque += r[0] * thrust[1] - r[1] * thrust[0]

        # Linear motion: F = ma
        accel = total_force / PAYLOAD_MASS

        # Apply Euler integration with linear damping.
        # v_next = (v_current + a * dt) * (1 - damping * dt)
        self.payload_vel = ((self.payload_vel + accel * DT) * 
                            (1.0 - LINEAR_DAMPING * DT))   
        self.payload_pos = self.payload_pos + self.payload_vel * DT
        
        # Angular motion
        ang_accel = total_torque / self.payload_inertia

        # ω_next = (ω_current + α * dt) * (1 - damping * dt)
        self.payload_omega = (
            (self.payload_omega + ang_accel * DT) * (1.0 - ANGULAR_DAMPING * DT)
        )
        self.payload_theta = self.payload_theta + self.payload_omega * DT

        self.timestep += 1    

        # Compute reward
        dist_to_goal = float(np.linalg.norm(self.payload_pos - END_POS))
        
        # wrap to [-pi, pi] so a full 360° spin reads as "level", not 2π off
        theta_err = abs((self.payload_theta - GOAL_THETA + np.pi) % (2*np.pi) - np.pi)
        reached_goal = (dist_to_goal < GOAL_RADIUS) and (theta_err < GOAL_THETA_TOL)

        # Penalize distance, angular velocity, and orientation error. Bonus for success.
        shared_reward = (-0.01 * dist_to_goal
                 - 0.001 * (self.payload_omega ** 2)
                 - 0.05 * (theta_err ** 2))  
         
        if reached_goal:
            shared_reward += 10.0
        rewards = {a: shared_reward for a in self.agents}

        # Check termination conditions
        terminations = {a: reached_goal for a in self.agents}

        # Check truncation conditions
        truncations = {a: self.timestep >= MAX_STEPS for a in self.agents}

        # Build observations and infos
        observations = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {"distance": dist_to_goal} for a in self.agents}

        # Clear agents when episode ends
        if reached_goal or self.timestep >= MAX_STEPS:
            self.agents = []

        # Render if required
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _drone_world_positions(self):
        """Return the (3, 2) array of drone positions in the world frame."""
        R = rot_matrix(self.payload_theta)
        return self.payload_pos[None, :] + self.triangle_local @ np.transpose(R)
    
    def _drone_world_velocities(self):
        """Return the (3, 2) array of drone velocities in the world frame.
        Each drone moves with the payload's translation velocity plus rotational
        contribution.
        """
        R = rot_matrix(self.payload_theta)
        world_offsets = self.triangle_local @ np.transpose(R)
        perp = np.stack([-world_offsets[:, 1], world_offsets[:, 0]], axis=1)
        return self.payload_vel[None, :] + self.payload_omega * perp


    def _get_obs(self, i):
        """Build a 17-dim vector for drone i."""
        drone_positions = self._drone_world_positions()
        drone_velocities = self._drone_world_velocities()

        own_pos = drone_positions[i]
        own_vel = drone_velocities[i]
        payload_rel = self.payload_pos - own_pos
        orient = np.array([np.cos(self.payload_theta), 
                           np.sin(self.payload_theta)], dtype=np.float32)

        other_indices = [(i + 1) % 3, (i + 2) % 3]
        others_rel = np.concatenate([drone_positions[j] - 
                                     own_pos for j in other_indices])
        
        goal_rel = END_POS - own_pos

        obs = np.concatenate([
            own_pos / ARENA_SIZE,                                   
            own_vel,                                                
            payload_rel / ARENA_SIZE,                                
            orient,                                                 
            self.payload_vel,                                        
            np.array([self.payload_omega], dtype=np.float32),        
            others_rel / ARENA_SIZE,                                 
            goal_rel / ARENA_SIZE,                                   
        ]).astype(np.float32)

        return obs

    def render(self):
        """Draw the current state with Pygame. Initializes the window on first call."""
        if self.render_mode is None:
            return

        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((600, 600))
            pygame.display.set_caption("SwarmLift")
            self._clock = pygame.time.Clock()

        self._screen.fill((30, 30, 40))   # dark background

        def to_screen(p):
            """Convert world coords [-ARENA_SIZE/2, +ARENA_SIZE/2] to pixel coords [0, 600]."""
            x = int((p[0] + ARENA_SIZE / 2) / ARENA_SIZE * 600)
            y = int(600 - (p[1] + ARENA_SIZE / 2) / ARENA_SIZE * 600)
            return (x, y)

        # Goal (green circle outline)
        pygame.draw.circle(self._screen, (80, 200, 80), to_screen(END_POS),
                           int(GOAL_RADIUS / ARENA_SIZE * 600), 2)

        # Payload triangle (filled blue, white outline)
        drone_positions = self._drone_world_positions()
        triangle_pts = [to_screen(p) for p in drone_positions]
        pygame.draw.polygon(self._screen, (100, 100, 160), triangle_pts)
        pygame.draw.polygon(self._screen, (200, 200, 255), triangle_pts, 2)

        # Drones (orange dots at corners)
        for p in drone_positions:
            pygame.draw.circle(self._screen, (255, 180, 80), to_screen(p), 6)

        # Payload centroid (small red dot)
        pygame.draw.circle(self._screen, (255, 100, 100), to_screen(self.payload_pos), 3)

        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

def naive_policy(observations):
    """Naive baseline: every drone pushes at full thrust toawward the goal (world frame)."""
    actions = {}
    for agent, obs in observations.items():
        # goal_rel is the last 2 entries of the observation, normalized by ARENA_SIZE
        goal_rel = obs[-2:] * ARENA_SIZE
        direction = goal_rel / (np.linalg.norm(goal_rel) + 1e-8)
        actions[agent] = (direction * MAX_THRUST).astype(np.float32)
    return actions


if __name__ == "__main__":
    import pygame

    env = SwarmLiftEnv(render_mode="human")
    obs, _ = env.reset(seed=0)

    episode = 1
    step_in_episode = 0
    total_reward = 0.0
    running = True

    while running:
        actions = naive_policy(obs)
        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += list(rewards.values())[0]
        step_in_episode += 1

        # Debug print every 20 steps
        if step_in_episode % 20 == 0:
            net_force = sum(actions[a] for a in env.agents)
            print(f"  step {step_in_episode:3d}: theta={env.payload_theta:+.2f} "
                  f"thrust0={actions['drone_0']} thrust1={actions['drone_1']} "
                  f"thrust2={actions['drone_2']} net_force={net_force}")
                  
        # If the episode ended, log it and start a new one
        if not env.agents:
            dist = infos[list(infos.keys())[0]]["distance"]
            outcome = "SUCCESS" if any(terms.values()) else "FAIL"
            print(f"Episode {episode}: {outcome}  steps={step_in_episode}  "
                  f"reward={total_reward:.2f}  final_dist={dist:.2f}")
            episode += 1
            step_in_episode = 0
            total_reward = 0.0
            obs, _ = env.reset(seed=episode)

    env.close()