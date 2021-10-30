from concurrent.futures import ThreadPoolExecutor

import os
import pickle
from PIL import Image, ImageTk
import time
from detector import convert_images
import numpy as np

from typing import List, Dict

from log_schema import Episode, Step

SCHEMA_VERSION = "1.0.0"


class Logger:
    def __init__(self, env, log_file):
        self.env = env
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count = 0

        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(4)
        # self.recording = []

    def log(self, step: Step, info: Dict):
        if self.episode.metadata is None:
            self.episode.metadata = info
        self.episode.steps.append(step)

    def reset_episode(self):
        self.episode = Episode(version=SCHEMA_VERSION)

    def on_episode_done(self):
        print(f"episode {self.episode_count} done, writing to file")

        # The next file cause all episodes to be written to the same pickle FP. (Overwrite first?)
        # self._multithreaded_recording.submit(lambda: self._commit(self.episode))
        self._commit(self.episode)
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count += 1

    def save_image_and_labels(self, episode):

        img_dir = "myapp"
        with open("my_app.txt", "a") as file:
            for step in episode.steps:
                t = str(round(time.time() * 1000))
                img_path = os.path.join(os.getcwd(), img_dir, t + ".png")

                img = step.obs
                steer, velocity = step.action
                file.write(t + " " + str(steer) + " " + str(velocity) + "\n")

                img_array = Image.fromarray(img)

                img_array.save(img_path)

        convert_images()

    def _commit(self, episode):
        # we use pickle to store our data
        # pickle.dump(self.recording, self._log_file)

        self.save_image_and_labels(episode)

        pickle.dump(episode, self._log_file)
        self._log_file.flush()
        # del self.recording[:]
        # self.recording.clear()

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()
