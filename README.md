# OvertakeAI: TrackMania AI with DQN

![Banner](https://github.com/user-attachments/assets/f748eba0-341e-48a9-9264-be26fb677b56)


This project uses `TMInterface` to control and train an AI agent for *TrackMania* using a Deep Q-Network (DQN). The AI learns optimal driving strategies by interacting with the game and receiving rewards for actions that improve performance, such as overtaking opponents or minimizing lap times.

## Features:
- **Game Control**: Uses `TMInterface` to send inputs (steering, throttle, brake) and gather game data (position, speed, checkpoints).
- **Reinforcement Learning**: Trains a DQN to decide the best action in any situation.
- **Performance Feedback**: Rewards are based on speed, checkpoint progress, and overtaking other cars.

## How It Works:
1. `TMInterface` connects to TrackMania to monitor the game and send commands.
2. The AI interacts with the environment, collecting data like position, speed, and track state.
3. The DQN model processes this data, predicts the best actions, and learns from feedback (reward signals).
4. Over time, the AI improves its driving to achieve better lap times and racing performance.

## Requirements:
- Python 3.x
- `TMInterface`
- TrackMania installation

## Development
- This project is currently not being maintained. I do not have a GPU anymore.
