import numpy as np
import pygame
import random
import matplotlib.pyplot as plt

# Constants for the grid and cell size
GRID_WIDTH, GRID_HEIGHT = 10, 7
CELL_SIZE = 50
SCREEN_WIDTH, SCREEN_HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Wind strengths for each column (negative values for upward wind)
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Windy Gridworld - Q-learning")

# Define the rewards
GOAL_REWARD = 100
GOAL2_REWARD = 200
STEP_REWARD = -1
BOUNDARY_REWARD = -10

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Exploration rate decay
EPSILON_DECAY = 0.995

# Maximum number of episodes
MAX_EPISODES = 5000

# Q-table to store action-values
Q_TABLE = np.zeros((GRID_WIDTH, GRID_HEIGHT, 4))  # 4 actions: up, down, left, right

# Agent starting position
agent_x, agent_y = 0, 3

# Goal positions
goal_x, goal_y = 7, 3
goal2_x, goal2_y = 8, 6

# Data collection lists
episode_rewards = []
episode_steps = []
episode_epsilons = []

def draw_grid():
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # Draw wind effect as arrows
            wind_strength = WIND[x]
            if wind_strength > 0:
                arrow_start = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE - 5)
                arrow_end = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + 5)
                arrow_color = BLUE if wind_strength == 1 else RED
                pygame.draw.line(screen, arrow_color, arrow_start, arrow_end, 3)

def draw_agent(x, y):
    agent_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, agent_rect)

def move_agent(action, x, y):
    if action == 0 and y > 0:
        y -= 1
    elif action == 1 and y < GRID_HEIGHT - 1:
        y += 1
    elif action == 2 and x > 0:
        x -= 1
    elif action == 3 and x < GRID_WIDTH - 1:
        x += 1
    return x, y

def apply_wind(x, y):
    if y > 0:  # Apply wind effect only when not at the top row
        y -= WIND[x]  # Subtract the wind strength of the corresponding column
        y = max(y, 0)  # Ensure y is not less than 0 (agent won't go above top row)
    return x, y

def get_reward(x, y):
    if x == goal_x and y == goal_y:
        return GOAL_REWARD
    elif x == goal2_x and y == goal2_y:
        return GOAL2_REWARD
    elif x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return BOUNDARY_REWARD
    else:
        return STEP_REWARD

def choose_action(x, y):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 3)  # Explore (random action)
    else:
        return np.argmax(Q_TABLE[x, y, :])  # Exploit (choose best action)

def main():
    global agent_x, agent_y, EPSILON  # Declare agent_x, agent_y, and EPSILON as global variables

    # Main game loop
    for episode in range(MAX_EPISODES):
        # Reset the agent's position to the starting position
        agent_x, agent_y = 0, 3

        # Variable to track if wind effect has been applied during the current move
        wind_applied = False

        # Episode-specific variables
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Choose an action using the epsilon-greedy policy
            action = choose_action(agent_x, agent_y)

            # Move the agent based on the chosen action
            new_agent_x, new_agent_y = move_agent(action, agent_x, agent_y)

            # Apply wind effect (only when moving upward)
            new_agent_x, new_agent_y = apply_wind(new_agent_x, new_agent_y)

            # Get the reward for the new state
            reward = get_reward(new_agent_x, new_agent_y)

            # Update the Q-table based on the Q-learning update rule
            old_q_value = Q_TABLE[agent_x, agent_y, action]
            max_q_value = np.max(Q_TABLE[new_agent_x, new_agent_y, :])
            new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q_value)
            Q_TABLE[agent_x, agent_y, action] = new_q_value

            # Update the agent's position based on the reward
            if reward != BOUNDARY_REWARD:  # Ignore boundary reward (agent doesn't move)
                agent_x, agent_y = new_agent_x, new_agent_y

            # Accumulate the total reward and number of steps for this episode
            total_reward += reward
            steps += 1

            # Draw the updated environment with the agent's position
            screen.fill(WHITE)
            draw_grid()
            draw_agent(agent_x, agent_y)

            # Draw the goal positions
            goal_rect = pygame.Rect(goal_x * CELL_SIZE, goal_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            goal2_rect = pygame.Rect(goal2_x * CELL_SIZE, goal2_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLUE, goal_rect)
            pygame.draw.rect(screen, BLUE, goal2_rect)

            # Update the display
            pygame.display.flip()

            # Add a time delay to slow down the agent's learning process
            pygame.time.delay(200)

            # Check if the agent reached the goal state or exceeded the maximum number of steps
            if agent_x == goal_x and agent_y == goal_y or agent_x == goal2_x and agent_y == goal2_y:
                done = True

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # Collect data for graphs
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_epsilons.append(EPSILON)

        # Decay the exploration rate (epsilon) after each episode
        EPSILON *= EPSILON_DECAY

        # Print the total reward obtained in this episode
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    # Quit Pygame
    pygame.quit()

    # Plot the graphs
    plot_episode_vs_reward(episode_rewards)
    plot_episode_vs_steps(episode_steps)
    plot_episode_vs_epsilon(episode_epsilons)

def plot_episode_vs_reward(rewards):
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode vs. Total Reward")
    plt.grid(True)
    plt.show()

def plot_episode_vs_steps(steps):
    plt.plot(range(1, len(steps) + 1), steps)
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.title("Episode vs. Number of Steps")
    plt.grid(True)
    plt.show()

def plot_episode_vs_epsilon(epsilons):
    plt.plot(range(1, len(epsilons) + 1), epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon (Exploration Rate)")
    plt.title("Episode vs. Epsilon")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
