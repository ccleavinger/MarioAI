import retro
import gym
from RandomAgent import TimeLimitWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class RetroWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info  # Add False for truncated

model = PPO.load("tmp/best_model.zip")

def main():
    steps = 0
    #env = retro.make(game='MegaMan2-Nes')
    env = retro.make(game='SuperMarioBros-Nes')
    env = RetroWrapper(env)  # Apply the custom wrapper
    env = TimeLimitWrapper(env)
    env = MaxAndSkipEnv(env, 4)

    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Final Info")
    print(info)
    env.close()


if __name__ == "__main__":
    main()