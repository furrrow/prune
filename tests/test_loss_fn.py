from unittest import TestCase

import torch
import numpy as np
from loss_fn import bradley_terry_loss


class Test(TestCase):

    def test_bt_equal(self):
        self.higher_score = torch.tensor(1.0)
        self.lower_score = torch.tensor(1.0)
        bt_score = bradley_terry_loss(self.higher_score, self.lower_score)
        # negative log score of 0.5
        score = 0.5
        neg_log_score = -torch.log(torch.tensor(score))
        print(f"equal, score {score:.4f} neg log loss {neg_log_score:.4f}")
        self.assertAlmostEqual(bt_score.item(), neg_log_score, places=3)

    def test_bt_high_low(self):
        self.higher_score = 1.0
        self.lower_score = -1.0
        bt_score = bradley_terry_loss(
            torch.tensor(self.higher_score), torch.tensor(self.lower_score))
        np_score = np.exp(self.higher_score) / (
            np.exp(self.higher_score) + np.exp(self.lower_score))
        neg_log_score = -np.log(np_score)
        print(f"high positive difference, score {np_score:.4f} neg log loss {neg_log_score:.4f}")
        self.assertAlmostEqual(bt_score.item(), neg_log_score, places=3)

    def test_bt_low_high(self):
        self.higher_score = -1.0
        self.lower_score = 1.0
        bt_score = bradley_terry_loss(
            torch.tensor(self.higher_score), torch.tensor(self.lower_score))
        np_score = np.exp(self.higher_score) / (
            np.exp(self.higher_score) + np.exp(self.lower_score))
        neg_log_score = -np.log(np_score)
        print(f"high neg difference, score {np_score:.4f} neg log loss {neg_log_score:.4f}")
        self.assertAlmostEqual(bt_score.item(), neg_log_score, places=3)
