"""
loss_fn.py
"""
import torch

def bradley_terry_loss(pref_score, rej_score):
    """
    -log(exp_pref / (exp_pref + exp_rej)) =  log(exp_pref + exp_rej) - log(exp_pref)
    the log(exp_pref) further simplifies into just the preference score
    :param pref_score:
    :param rej_score:
    :return:
    """
    log_numerator = pref_score
    log_denominator = torch.log(torch.exp(pref_score) + torch.exp(rej_score))
    loss = log_denominator - log_numerator
    return loss.mean()


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