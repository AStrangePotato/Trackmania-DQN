import os
import gym
import numpy as np
import torch
from collections import deque
from ppo import PPO, Memory
import cv2

CHECKPOINT_PATH = "ppo_carracing.pth"

def stack_frames(frames, state):
    frames.append(state)
    if len(frames) < 4:
        for _ in range(4 - len(frames)):
            frames.append(state)
    stacked_state = np.stack(frames, axis=0)
    return stacked_state, frames

def preprocess_frame(frame):
    assert isinstance(frame, np.ndarray), f"Frame is not np.ndarray, got {type(frame)}"
    frame = cv2.resize(frame, (96, 96))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame

def save_checkpoint(ppo, memory, episode, timestep):
    checkpoint = {
        'policy_state_dict': ppo.policy.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'episode': episode,
        'timestep': timestep,
        'memory': memory.__dict__,  # Save memory contents
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(ppo, memory):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        ppo.policy.load_state_dict(checkpoint['policy_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        memory.__dict__.update(checkpoint['memory'])
        episode = checkpoint['episode']
        timestep = checkpoint['timestep']
        print(f"Checkpoint loaded from episode {episode}")
        return episode, timestep
    return 1, 0

def main():
    env = gym.make('CarRacing-v2', render_mode=None)
    action_dim = env.action_space.shape[0]
    ppo = PPO(action_dim)
    memory = Memory()
    max_episodes = 1000
    max_timesteps = 1000
    update_timestep = 4000
    timestep = 0

    # Load checkpoint if available
    start_episode, timestep = load_checkpoint(ppo, memory)

    for episode in range(start_episode, max_episodes + 1):
        state, _ = env.reset()
        state = preprocess_frame(state)
        frames = deque(maxlen=4)
        stacked_state, frames = stack_frames(frames, state)
        episode_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0).cuda()
            action, log_prob, value = ppo.policy.act(state_tensor)
            action_np = action.cpu().detach().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            next_state = preprocess_frame(next_state)
            stacked_next_state, frames = stack_frames(frames, next_state)

            memory.states.append(state_tensor.squeeze(0))
            memory.actions.append(action.squeeze(0))
            memory.log_probs.append(log_prob)
            memory.rewards.append(reward)
            memory.dones.append(done)
            memory.values.append(value)

            episode_reward += reward
            stacked_state = stacked_next_state

            if timestep % update_timestep == 0:
                with torch.no_grad():
                    last_state_tensor = torch.tensor(stacked_state, dtype=torch.float32).unsqueeze(0).cuda()
                    _, _, last_value = ppo.policy.act(last_state_tensor)
                memory.returns = ppo.compute_returns(memory.rewards, memory.dones, last_value)
                ppo.update(memory)
                memory.clear()
                save_checkpoint(ppo, memory, episode, timestep)
                timestep = 0

            if done:
                break

        print(f"Episode {episode} Reward: {episode_reward}")

        # Save checkpoint after each episode as well
        save_checkpoint(ppo, memory, episode, timestep)

    env.close()

if __name__ == "__main__":
    main()
