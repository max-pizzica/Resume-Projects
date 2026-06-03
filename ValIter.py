import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os

# Parameters
TRACK_NAME = ".venv/U-track.txt"
CRASH_POS = "STRT" #STRT or NRST, denotes whether car resets at start or at nearest position following a crash


class RaceTrack:
    def __init__(self, filename):
        self.load_track(filename)
        self.max_velocity = 5
        self.action_fail_prob = 0.2
        # All possible actions (ax, ay)
        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

    def load_track(self, filename):
        """Load track from file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        # Parse dimensions
        self.rows, self.cols = map(int, lines[0].strip().split(','))
        # Parse track grid
        self.grid = []
        self.start_positions = []
        self.finish_positions = []
        for i in range(1, self.rows + 1):
            row = list(lines[i].strip())
            self.grid.append(row)
            for j, cell in enumerate(row):
                if cell == 'S':
                    self.start_positions.append((i - 1, j))
                elif cell == 'F':
                    self.finish_positions.append((i - 1, j))
        self.grid = np.array(self.grid)
        print(f"Track loaded: {self.rows}x{self.cols}")
        print(f"Start positions: {len(self.start_positions)}")
        print(f"Finish positions: {len(self.finish_positions)}")

    def is_valid_position(self, x, y):
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return False
        return self.grid[x, y] != '#'

    def is_finish_line(self, x, y):
        return self.grid[x, y] == 'F'

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        # Handle case where vx,vy = 0
        if x0 == x1 and y0 == y1:
            return [(x0, y0)]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        # Determine which axis is the driving axis
        if dx > dy:
            # X is driving axis
            error = dx / 2.0
            while x != x1:
                points.append((x, y))
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
            points.append((x, y))  # Add final point
        else:
            # Y is driving axis
            error = dy / 2.0
            while y != y1:
                points.append((x, y))
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
            points.append((x, y))  # Add final point
        return points

    def get_next_state(self, x, y, vx, vy, ax, ay, action_succeeds, crash_mode='NRST'):
        # Update velocity
        if action_succeeds:
            new_vx = np.clip(vx + ax, -self.max_velocity, self.max_velocity)
            new_vy = np.clip(vy + ay, -self.max_velocity, self.max_velocity)
        else:
            new_vx = vx
            new_vy = vy
        # Calculate new position
        new_x = x + new_vx
        new_y = y + new_vy
        path = self.bresenham_line(x, y, new_x, new_y)
        for i in range(1, len(path)):
            px, py = path[i]
            if self.is_finish_line(px, py):
                return (px, py, new_vx, new_vy, False, True)
            # Check if hit wall
            if not self.is_valid_position(px, py):
                if crash_mode == 'NRST':
                    # Return to previous valid position
                    prev_x, prev_y = path[i - 1]
                    return (prev_x, prev_y, 0, 0, True, False)
                else:  # STRT mode
                    # Will be handled by returning special flag
                    return (-1, -1, 0, 0, True, False)
        return (new_x, new_y, new_vx, new_vy, False, False)

    def visualize_track(self, trajectory=None):
        fig, ax = plt.subplots(figsize=(12, 10))
        track_array = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == '#':
                    track_array[i, j] = [0.1, 0.1, 0.1]  # Wall - dark gray
                elif self.grid[i, j] == 'S':
                    track_array[i, j] = [0, 1, 0]  # Start - green
                elif self.grid[i, j] == 'F':
                    track_array[i, j] = [1, 0, 0]  # Finish - red
                else:
                    track_array[i, j] = [0.7, 0.7, 0.7]  # Track - light gray
        ax.imshow(track_array)
        # Plot trajectory
        if trajectory:
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2, alpha=0.6)
            ax.plot(trajectory[0, 1], trajectory[0, 0], 'g*', markersize=10, label='Start')
            ax.plot(trajectory[-1, 1], trajectory[-1, 0], 'r*', markersize=15, label='End')
            ax.legend()
        ax.set_title('Race Track')
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        plt.tight_layout()
        return fig

class ValueIteration:
    def __init__(self, track, discount_factor=0.9, convergence_threshold=0.01, crash_mode='NRST'):
        self.track = track
        self.gamma = discount_factor
        self.threshold = convergence_threshold
        self.crash_mode = crash_mode
        # Initialize value function and policy
        self.V = defaultdict(float)
        self.policy = {}
        print(f"Value Iteration initialized (γ={self.gamma}, θ={self.threshold}, crash_mode={crash_mode})")

    def get_all_states(self):
        #generate all possible states for MDP
        states = []
        for x in range(self.track.rows):
            for y in range(self.track.cols):
                if not self.track.is_valid_position(x, y):
                    continue
                for vx in range(-self.track.max_velocity, self.track.max_velocity + 1):
                    for vy in range(-self.track.max_velocity, self.track.max_velocity + 1):
                        states.append((x, y, vx, vy))
        return states

    def train(self, max_iterations=1000):
        print(f"\nStarting Value Iteration training...")
        print(f"Generating state space...")
        states = self.get_all_states()
        print(f"State space size: {len(states)}")
        iteration_times = []
        deltas = []
        for iteration in range(max_iterations):
            start_time = time.time()
            delta = 0
            # Update value for each state
            for state in states:
                x, y, vx, vy = state
                # Skip terminal states (finish line)
                if self.track.is_finish_line(x, y):
                    continue
                old_value = self.V[state]
                action_values = []
                for action in self.track.actions:
                    ax, ay = action
                    value = self.compute_action_value(x, y, vx, vy, ax, ay)
                    action_values.append((value, action))
                # Update value and policy
                best_value, best_action = max(action_values, key=lambda x: x[0])
                self.V[state] = best_value
                self.policy[state] = best_action
                delta = max(delta, abs(old_value - best_value))
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
            deltas.append(delta)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: delta={delta:.6f}, time={iteration_time:.2f}s")
            # Check for convergence
            if delta < self.threshold:
                print(f"\nConverged after {iteration + 1} iterations!")
                break
        print(f"Training complete. Total iterations: {iteration + 1}")
        return deltas, iteration_times

    def compute_action_value(self, x, y, vx, vy, ax, ay):
        value = 0
        # Action succeeds with probability 0.8
        next_state_success = self.track.get_next_state(
            x, y, vx, vy, ax, ay, True, self.crash_mode
        )
        value += 0.8 * self.get_state_value(next_state_success)
        # Action fails with probability 0.2 (no acceleration)
        next_state_fail = self.track.get_next_state(
            x, y, vx, vy, 0, 0, False, self.crash_mode
        )
        value += 0.2 * self.get_state_value(next_state_fail)
        return value

    def get_state_value(self, state_info):
        new_x, new_y, new_vx, new_vy, crashed, finished = state_info
        # Reward = -1 for each step
        reward = -1
        if finished: #terminal state
            return reward
        elif crashed and self.crash_mode == 'STRT':
            # Large penalty for crashing and restarting
            return reward - 100
        else:
            # Regular transition
            next_state = (new_x, new_y, new_vx, new_vy)
            return reward + self.gamma * self.V[next_state]

    def simulate_race(self, start_position=None, max_steps=1000, verbose=False):
        if start_position is None:
            start_position = self.track.start_positions[
                np.random.randint(len(self.track.start_positions))
            ]
        x, y = start_position
        vx, vy = 0, 0
        trajectory = [(x, y)]
        steps = 0
        while steps < max_steps:
            state = (x, y, vx, vy)
            # Get action from policy
            if state in self.policy:
                ax, ay = self.policy[state]
            else:
                # Default action if state not in policy
                ax, ay = (0, 0)
            # Stochastic action execution
            action_succeeds = np.random.random() > self.track.action_fail_prob
            # Get next state
            new_x, new_y, new_vx, new_vy, crashed, finished = self.track.get_next_state(
                x, y, vx, vy, ax, ay, action_succeeds, self.crash_mode
            )
            if finished:
                trajectory.append((new_x, new_y))
                if verbose:
                    print(f"Finished in {steps + 1} steps!")
                return trajectory, steps + 1, True
            if crashed and self.crash_mode == 'STRT':
                # Restart from beginning
                x, y = start_position
                vx, vy = 0, 0
                if verbose:
                    print(f"Crashed at step {steps}, restarting...")
            else:
                x, y = new_x, new_y
                vx, vy = new_vx, new_vy
            trajectory.append((x, y))
            steps += 1
        if verbose:
            print(f"Did not finish within {max_steps} steps")
        return trajectory, steps, False


class QLearning:
    def __init__(self, track, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.1, crash_mode='NRST'):
        self.track = track
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.crash_mode = crash_mode
        # Initialize Q-table
        self.Q = defaultdict(lambda: defaultdict(float))
        print(f"Q-Learning initialized (α={self.alpha}, γ={self.gamma}, ε={self.epsilon}, crash_mode={crash_mode})")

    def get_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return self.track.actions[np.random.randint(len(self.track.actions))]
        else:
            # Exploit: best action
            x, y, vx, vy = state
            action_values = []
            for action in self.track.actions:
                value = self.Q[state][action]
                action_values.append((value, action))
            # Return action with highest Q-value (random tie-breaking)
            max_value = max(action_values, key=lambda x: x[0])[0]
            best_actions = [a for v, a in action_values if v == max_value]
            return best_actions[np.random.randint(len(best_actions))]

    def train(self, num_episodes=1000, max_steps_per_episode=1000,
              decay_epsilon=True, verbose_every=100):
        print(f"\nStarting Q-Learning training...")
        print(f"Number of episodes: {num_episodes}")
        episode_rewards = []
        episode_steps = []
        episode_finished = []
        initial_epsilon = self.epsilon
        for episode in range(num_episodes):
            # Decay epsilon over time
            if decay_epsilon:
                self.epsilon = initial_epsilon * (1 - episode / num_episodes)
            # Start from random start position
            start_pos = self.track.start_positions[
                np.random.randint(len(self.track.start_positions))
            ]
            x, y = start_pos
            vx, vy = 0, 0
            episode_reward = 0
            steps = 0
            finished = False
            for step in range(max_steps_per_episode):
                state = (x, y, vx, vy)
                # Choose action
                action = self.get_action(state, training=True)
                ax, ay = action
                # Stochastic action
                action_succeeds = np.random.random() > self.track.action_fail_prob
                # Take action and observe next state
                new_x, new_y, new_vx, new_vy, crashed, finished = self.track.get_next_state(
                    x, y, vx, vy, ax, ay, action_succeeds, self.crash_mode
                )
                # Reward
                reward = -1
                # Handle terminal states
                if finished:
                    next_state = (new_x, new_y, new_vx, new_vy)
                    # Terminal state - no future value
                    target = reward
                elif crashed and self.crash_mode == 'STRT':
                    # Restart from beginning with large penalty
                    reward = -101  # -1 step cost + -100 crash penalty
                    new_x, new_y = start_pos
                    new_vx, new_vy = 0, 0
                    next_state = (new_x, new_y, new_vx, new_vy)
                    # Continue episode
                    target = reward + self.gamma * max(
                        self.Q[next_state][a] for a in self.track.actions
                    )
                else:
                    next_state = (new_x, new_y, new_vx, new_vy)
                    # Regular transition
                    target = reward + self.gamma * max(
                        self.Q[next_state][a] for a in self.track.actions
                    )
                # Q-Learning update
                old_q = self.Q[state][action]
                self.Q[state][action] = old_q + self.alpha * (target - old_q)
                # Update state
                x, y = new_x, new_y
                vx, vy = new_vx, new_vy
                episode_reward += reward
                steps += 1
                if finished:
                    break
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            episode_finished.append(finished)
            if (episode + 1) % verbose_every == 0:
                recent_rewards = episode_rewards[-verbose_every:]
                recent_finished = episode_finished[-verbose_every:]
                print(f"Episode {episode + 1}: avg_reward={np.mean(recent_rewards):.2f}, "
                      f"success_rate={np.mean(recent_finished):.2%}, ε={self.epsilon:.3f}")
        print(f"\nTraining complete!")
        return episode_rewards, episode_steps, episode_finished

    def simulate_race(self, start_position=None, max_steps=1000, verbose=False):
        if start_position is None:
            start_position = self.track.start_positions[
                np.random.randint(len(self.track.start_positions))
            ]
        x, y = start_position
        vx, vy = 0, 0
        trajectory = [(x, y)]
        steps = 0
        while steps < max_steps:
            state = (x, y, vx, vy)
            # Get action (greedy, no exploration)
            action = self.get_action(state, training=False)
            ax, ay = action
            # Stochastic action execution
            action_succeeds = np.random.random() > self.track.action_fail_prob
            # Get next state
            new_x, new_y, new_vx, new_vy, crashed, finished = self.track.get_next_state(
                x, y, vx, vy, ax, ay, action_succeeds, self.crash_mode
            )
            if finished:
                trajectory.append((new_x, new_y))
                if verbose:
                    print(f"Finished in {steps + 1} steps!")
                return trajectory, steps + 1, True
            if crashed and self.crash_mode == 'STRT':
                # Restart from beginning
                x, y = start_position
                vx, vy = 0, 0
                if verbose:
                    print(f"Crashed at step {steps}, restarting...")
            else:
                x, y = new_x, new_y
                vx, vy = new_vx, new_vy
            trajectory.append((x, y))
            steps += 1
        if verbose:
            print(f"Did not finish within {max_steps} steps")

        return trajectory, steps, False


def run_value_iteration(track_file, num_trials=10, crash_mode='NRST'):
    print(f"Running VALUE ITERATION on {track_file}")
    print(f"Crash mode: {crash_mode}")
    # Load track
    track = RaceTrack(track_file)
    # Train with Value Iteration
    vi = ValueIteration(track, discount_factor=0.95, convergence_threshold=0.0001, crash_mode=crash_mode)
    deltas, iteration_times = vi.train(max_iterations=1000)
    # Plot learning curve
    # Run simulation trials
    print(f"\nRunning {num_trials} simulation trials...")
    results = []
    for trial in range(num_trials):
        trajectory, steps, finished = vi.simulate_race(verbose=False)
        results.append({
            'trial': trial + 1,
            'steps': steps,
            'finished': finished,
            'trajectory_length': len(trajectory)
        })
        print(f"Trial {trial + 1}: {'Finished' if finished else 'Did not finish'} - {steps} steps")
    # Calculate statistics
    finished_trials = [r for r in results if r['finished']]
    if finished_trials:
        steps_list = [r['steps'] for r in finished_trials]
        print(f"\n{'=' * 60}")
        print(f"Results Summary:")
        print(f"  Successful trials: {len(finished_trials)}/{num_trials}")
        print(f"  Mean steps: {np.mean(steps_list):.2f}")
        print(f"  Std steps: {np.std(steps_list):.2f}")
        print(f"  Min steps: {np.min(steps_list)}")
        print(f"  Max steps: {np.max(steps_list)}")
        print(f"{'=' * 60}\n")
    else:
        print("\nNo trials finished successfully!")
    # Visualize one successful trajectory
    if finished_trials:
        print("Generating trajectory visualization...")
        trajectory, steps, _ = vi.simulate_race(verbose=True)
        fig = track.visualize_track(trajectory)
        fig.savefig(f'GROUP_2_ValItr_{TRACK_NAME}{crash_mode}.png')
        print(f"Trajectory plot saved to GROUP_2_ValItr_{TRACK_NAME}{crash_mode}.png")
        plt.close(fig)
    return vi, results


def run_q_learning(track_file, num_episodes=5000, num_trials=10, crash_mode='NRST'):
    print(f"Running Q-LEARNING on {track_file}")
    print(f"Crash mode: {crash_mode}")
    # Load track
    track = RaceTrack(track_file)
    # Train with Q-Learning
    ql = QLearning(track, learning_rate=0.25, discount_factor=0.9,
                   epsilon=0.3, crash_mode=crash_mode)
    episode_rewards, episode_steps, episode_finished = ql.train(
        num_episodes=num_episodes,
        max_steps_per_episode=1000,
        decay_epsilon=True,
        verbose_every=500
    )
    # Run simulation trials
    print(f"\nRunning {num_trials} simulation trials...")
    results = []
    for trial in range(num_trials):
        trajectory, steps, finished = ql.simulate_race(verbose=False)
        results.append({
            'trial': trial + 1,
            'steps': steps,
            'finished': finished,
            'trajectory_length': len(trajectory)
        })
        print(f"Trial {trial + 1}: {'Finished' if finished else 'Did not finish'} - {steps} steps")
    # Calculate statistics
    finished_trials = [r for r in results if r['finished']]
    if finished_trials:
        steps_list = [r['steps'] for r in finished_trials]
        print(f"\n{'=' * 60}")
        print(f"Results Summary:")
        print(f"  Successful trials: {len(finished_trials)}/{num_trials}")
        print(f"  Mean steps: {np.mean(steps_list):.2f}")
        print(f"  Std steps: {np.std(steps_list):.2f}")
        print(f"  Min steps: {np.min(steps_list)}")
        print(f"  Max steps: {np.max(steps_list)}")
        print(f"{'=' * 60}\n")
    else:
        print("\nNo trials finished successfully!")
    # Visualize one successful trajectory
    if finished_trials:
        print("Generating trajectory visualization...")
        trajectory, steps, _ = ql.simulate_race(verbose=True)
        fig = track.visualize_track(trajectory)
        fig.savefig(f'GROUP_2_QLearning_{TRACK_NAME}{crash_mode}.png')
        print(f"Trajectory plot saved to GROUP_2_QLearning_{TRACK_NAME}{crash_mode}.png")
        plt.close(fig)
    return ql, results


if __name__ == "__main__":
        TRACK_NAME = os.path.basename(TRACK_NAME)
        agent, results = run_value_iteration(
            TRACK_NAME,
            num_trials=10,
            crash_mode=CRASH_POS
        )


        print("\nAll experiments complete!")