# --------------------------------------------------------------------
# MultiModal Embodied Tracking
#  - Random model
# --------------------------------------------------------------------

import numpy as np

class RandomModel:
    def __init__(
        self, 
        action_num: int = 4,
        action_prob: list = [0.01, 0.79, 0.10, 0.10]
    ) -> None:
        assert len(action_prob) == action_num

        self.action_num = action_num
        self.action_prob = action_prob

    def predict(self, image, depth=None):
        """
        Get the action based on the action_prob
            stop = 0
            move_forward = 1
            turn_left = 2
            turn_right = 3
        """
        action = np.random.choice(self.action_num, p=self.action_prob)
        return action