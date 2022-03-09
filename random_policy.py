from utils.make_env import arg_parse, get_env_kwargs, make_env
import matplotlib.pyplot as plt
import imageio
import os

def main():
    args = arg_parse()
    env_kwargs = get_env_kwargs(args)
    env = make_env(args.env, 0, args.log_path, done_when_success=True, flatten_dict=True, kwargs=env_kwargs)
    obs = env.reset()
    done = False

    video_name = "video.mp4"
    folder = os.path.dirname(os.path.abspath(__file__))
    writer = imageio.get_writer(os.path.join(folder, video_name), fps=10)
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(mode="rgb_array")
        writer.append_data(img)
        plt.pause(0.1)

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
