from utils.make_env import arg_parse, get_env_kwargs, make_env
import matplotlib.pyplot as plt
import imageio
import os
import time
import numpy as np

actions = []
observations = []
infos = []

images = []  # record video

def main():
    args = arg_parse()
    env_kwargs = get_env_kwargs(args)
    num_itr = 10
    cnt = 0
    init_state_space = 'random'
    folder = os.path.dirname(os.path.abspath(__file__))

    env = make_env(args.env, 0, args.log_path, done_when_success=True, flatten_dict=False, kwargs=env_kwargs)
    print("Reset!")
    init_time = time.time()

    print()
    while len(actions) < num_itr:
        obs = env.reset()
        print("ITERATION NUMBER ", len(actions))
        goToGoal(env, obs)
        cnt += 1

    file_name = "data/data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += ".npz"

    np.savez_compressed(os.path.join(folder, file_name),
                        acs=actions, obs=observations, info=infos)  # save the file

    obs = env.reset()
    done = False
    video_name = "video.mp4"
    writer = imageio.get_writer(os.path.join(folder, video_name), fps=10)
    while not done:
        #action = env.action_space.sample()
        action = env.get_oracle_action(obs)
        obs, reward, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        writer.append_data(img)
        plt.pause(0.1)

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")

    writer.close()
    env.close()

def goToGoal(env, last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    episode_obs.append(last_obs)

    obs, success = last_obs, False

    while time_step < 50:
        action = env.get_oracle_action(obs)
        obs, reward, done, info = env.step(action)
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        episode_acs.append(action)
        episode_info.append(info)
        episode_obs.append(obs)
    print(success)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    if success:
        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)


if __name__ == "__main__":
    main()