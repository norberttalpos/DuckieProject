import argparse
import os

import numpy as np
from gym_duckietown.envs import DuckietownEnv
from keras.models import load_model

from duckieGym.DaggerLearner import DaggerLearner
from duckieGym.DaggerTeacher import DaggerTeacher
from duckieGym.IIL import InteractiveImitationLearning
from duckieGym.daggerSandBox import MyInteractiveImitationLearning
from duckieGym.detector import preprocess_image


class DAgger(MyInteractiveImitationLearning):
    """
    DAgger algorithm to mix policies between learner and expert
    Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." Proceedings of the fourteenth international conference on artificial intelligence and statistics. 2011.
    ...
    Methods
    -------
    _mix
        used to return a policy teacher / expert based on random choice and safety checks
    """

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5, test=False):
        MyInteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, test)
        # expert decay
        self.p = alpha
        self.alpha = self.p

        self.teacherDecisions = 0
        self.learnerDecisions = 0
        self.learnerIdxs = []  # frame indexes during which the learner was behind the wheel

        # thresholds used to give control back to learner once the teacher converges
        self.convergence_distance = 0.05
        self.convergence_angle = np.pi / 18

        # threshold on angle and distance from the lane when using the model to avoid going off track and env reset within an episode
        self.angle_limit = np.pi / 8
        self.distance_limit = 0.12

    def _mix(self):

        control_policy = self.learner  #swapped from: np.random.choice(a=[self.teacher, self.learner], p=[self.alpha, 1.0 - self.alpha])

        if self._found_obstacle:
            return self.teacher
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except:
            return control_policy
        if self.active_policy:
            # keep using teacher until duckiebot converges back on track
            if not (abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad) < self.convergence_angle):
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give
            # control back to the expert
            if abs(lp.dist) > self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                return self.teacher

        self.learnerIdxs.append(self.learnerDecisions + self.teacherDecisions)
        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)

    @property
    def observations(self):
        return self._observations


if __name__ == "__main__":
    # ! Parser sector:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="zigzag_dists")
    parser.add_argument(
        "--draw-curve", default=False, help="draw the lane following curve"
    )
    parser.add_argument(
        "--draw-bbox", default=False, help="draw collision detection bounding boxes"
    )
    parser.add_argument(
        "--domain-rand", default=False, help="enable domain randomization"
    )
    parser.add_argument("--distortion", default=True)

    parser.add_argument(
        "--raw-log", default=False, help="enables recording high resolution raw log"
    )
    parser.add_argument(
        "--steps", default=1000, help="number of steps to record in one batch", type=int
    )
    parser.add_argument("--nb-episodes", default=1, type=int)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--downscale", action="store_true")

    args = parser.parse_args()

    # ! Start Env

    env = DuckietownEnv(
        map_name=args.map_name,
        max_steps=args.steps,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        distortion=args.distortion,
        accept_start_angle_deg=4,
        full_transparency=True,
    )

    model = load_model("0.3val_loss.hdf5")
    iil = DAgger(env=env, teacher=DaggerTeacher(env), learner=DaggerLearner(model), horizon=500, episodes=1)
    iil.train()
    observation = iil.get_observations()

    for id, obs in enumerate(observation):
        img = preprocess_image(obs)

        path = os.path.join(os.getcwd(), "daggerObservations")
        img.save(os.path.join(path, str(id) + ".png"))

    labels = iil.get_expert_actions()

    filepath = os.path.join(os.getcwd(), "daggerObservations")
    with open(os.path.join(os.getcwd(), "labels.txt"), "w") as f:
        for label in labels:
            f.write(str(label[0]) + " " + str(label[1]))
            f.write("\n")
