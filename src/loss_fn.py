"""
loss_fn.py
"""
import torch
import torch.nn.functional as F


def bradley_terry_loss(pref_score, rej_score):
    """
    P(i > j) = sigmoid(s_i - s_j)
    Loss = -log(sigmoid(s_i - s_j))
    Using logsigmoid for numerical stability
    """
    return -F.logsigmoid(pref_score - rej_score).mean()


class PL_Loss(torch.nn.Module):
    def forward(self, rewards, pref_idx):
        """
        Args:
            rewards: Tensor of shape (batch_size, num_actions) with unordered predicted rewards.
            pref_idx: Tensor of shape (batch_size, num_actions), contains indices that define the correct ranking.

        Returns:
            Scalar loss value.
        """

        ordered_rewards = torch.gather(rewards, 1, pref_idx[:, :, 0])  # Align with true preference ranking

        # Compute PL Loss
        log_denominators = torch.logcumsumexp(ordered_rewards.flip(dims=[1]), dim=1).flip(dims=[1])
        loss = ordered_rewards - log_denominators
        loss = -loss[:, :-1].sum(dim=1)

        return loss.mean()  # Average over batch
