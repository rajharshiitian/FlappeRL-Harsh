FlaapRL: Reinforcement Learning Agent for Flappy Bird | SOC Project


This repository hosts the code and documentation for FlaapRL, a reinforcement learning-based agent developed to autonomously play the Flappy Bird game. The project was undertaken as part of the Student Open Curriculum (SOC) program, focusing on applying reinforcement learning (RL) techniques to game environments.

Project Overview
Flappy Bird is a popular 2D arcade game where the player controls a bird navigating through a series of pipes without crashing. The game's simple yet challenging nature makes it an excellent testbed for reinforcement learning algorithms. FlaapRL aims to train an AI agent capable of achieving high scores in Flappy Bird by learning optimal actions through trial and error.

Key Features
Environment Setup: Integrated the Flappy Bird game environment using the PyGame framework, enabling seamless interaction between the game and the RL agent.
Reinforcement Learning Algorithm: Implemented the Deep Q-Network (DQN) algorithm, allowing the agent to learn the best strategies by maximizing cumulative rewards.
Training Process: The agent was trained over multiple episodes, with Q-learning used to update the policy based on the rewards received for each action. Techniques like experience replay and target networks were employed to stabilize training.
Performance Evaluation: Evaluated the agent's performance by tracking its scores over time and comparing them against baseline performance (random actions). Visualized the learning curve to demonstrate the improvement in gameplay.
Results
FlaapRL successfully learned to navigate the bird through pipes, achieving scores significantly higher than random actions after sufficient training. The agent displayed an improved understanding of the game's dynamics, such as when to flap to avoid obstacles.

Future Enhancements
Implementing advanced RL algorithms like Double DQN or Proximal Policy Optimization (PPO) for further performance improvements.
Experimenting with different reward structures to encourage more strategic gameplay.
Extending the project to other similar arcade games to test the generalization capability of the RL agent.
Technologies Used
Python
TensorFlow/Keras
PyGame
OpenAI Gym (custom environment)
Matplotlib
