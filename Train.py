import os

import gym
import numpy as np
import optuna
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor

from RandomAgent import TimeLimitWrapper


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


class RetroWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info  # Add False for truncated


def make_env(env_id, rank, seed=0):
    def _init():
        env = retro.make(game=env_id)
        env = RetroWrapper(env)  # Apply the custom wrapper
        env = TimeLimitWrapper(env, max_steps=2000)
        env = MaxAndSkipEnv(env, 4)
        return env

    set_random_seed(seed)
    return _init

def create_env(env_id, num_cpu):
    return VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), "tmp/TestMonitor")

# optuna training objective
def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 16, 2048)
    batch_size = trial.suggest_int('batch_size', 8, 256)
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-1)

    # Create the environment
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4
    env = create_env(env_id, num_cpu)

    # Create the model with the suggested hyperparameters
    model = PPO('CnnPolicy', env, verbose=0, tensorboard_log="./board/",
                learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
                clip_range=clip_range, ent_coef=ent_coef)

    # Create an EvalCallback for evaluation during training
    eval_env = create_env(env_id, 1)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1_000,
                                 deterministic=True, render=False)

    # Train the model
    try:
        model.learn(total_timesteps=10_0000, callback=eval_callback)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return float('-inf')

    # Return the negative of the mean reward (we want to maximize reward)
    return -eval_callback.best_mean_reward

if __name__ == '__main__':
    # Create an Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    # Print the best hyperparameters and score
    print("Best hyperparameters:", study.best_params)
    print("Best score:", -study.best_value)

    # Train the final model with the best hyperparameters
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4
    env = create_env(env_id, num_cpu)

    best_params = study.best_params
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", **best_params)

    print("------------- Start Learning with Best Hyperparameters -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir="tmp/")
    model.learn(total_timesteps=500_000, callback=callback, tb_log_name="PPO-best")
    model.save(env_id)
    print("------------- Done Learning -------------")

    # Test the trained model
    env = retro.make(game=env_id)
    env = RetroWrapper(env)
    env = TimeLimitWrapper(env)

    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
