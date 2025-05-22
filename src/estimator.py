"""Off-Policy Estimators for Slate/Ranking Policies."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from obp.utils import check_sips_inputs
from obp.utils import estimate_confidence_interval_by_bootstrap
from obp.ope.estimators_slate import(
    BaseSlateOffPolicyEstimator,
)






@dataclass
class BaseSlateInverseProbabilityWeighting(BaseSlateOffPolicyEstimator):
    """Base Class of Inverse Probability Weighting Estimators for the slate contextual bandit setting.

    len_list: int (> 1)
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, `len_list=3`.

    """

    len_list: int

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate rewards for each slate and slot (position).

        Parameters
        ----------
        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        behavior_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of behavior policy choosing a particular action at each position (slot) or
            joint probabilities of behavior policy choosing a whole slate/ranking of actions.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Marginal probabilities of evaluation policy choosing a particular action at each slot/position or
            joint probabilities of evaluation policy choosing a whole slate/ranking of actions.

        Returns
        ----------
        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

        """
        iw = evaluation_policy_pscore / behavior_policy_pscore
        estimated_rewards = reward * iw
        return estimated_rewards

    def _estimate_slate_confidence_interval_by_bootstrap(
        self,
        slate_id: np.ndarray,
        estimated_rewards: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        estimated_rewards: array-like, shape (<= n_rounds * len_list,)
            Rewards estimated for each slate and slot (position).

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        unique_slate = np.unique(slate_id)
        # sum estimated_rewards in each slate
        estimated_round_rewards = list()
        for slate in unique_slate:
            estimated_round_rewards.append(estimated_rewards[slate_id == slate].sum())
        estimated_round_rewards = np.array(estimated_round_rewards)
        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class ClickBasedIPS(BaseSlateInverseProbabilityWeighting):
    """Standard Inverse Propensity Scoring (SIPS) Estimator.

    Note
    -------
    SIPS estimates the policy value of evaluation policy :math:`\\pi_e`
    without imposing any assumption on user behavior.
    Please refer to Eq.(1) in Section 3 of McInerney et al.(2020) for more details.

    Parameters
    ----------
    estimator_name: str, default='sips'.
        Name of the estimator.

    References
    ------------
    James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Ben Carterette.
    "Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions.", 2020.

    """

    estimator_name: str = "cips"

    def estimate_policy_value(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        behavior_policy_p_click: np.ndarray,
        evaluation_policy_p_click: np.ndarray,
        **kwargs,
    ) -> float:
        """Estimate the policy value of evaluation policy.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        # check_sips_inputs(
        #     slate_id=slate_id,
        #     reward=reward,
        #     position=position,
        #     pscore=behavior_policy_p_click,
        #     evaluation_policy_pscore=evaluation_policy_p_click,
        # )
        return (
            self._estimate_round_rewards(
                reward=reward,
                position=position,
                behavior_policy_pscore=behavior_policy_p_click,
                evaluation_policy_pscore=evaluation_policy_p_click,
            ).sum()
            / np.unique(slate_id).shape[0]
        )

    def estimate_interval(
        self,
        slate_id: np.ndarray,
        reward: np.ndarray,
        position: np.ndarray,
        pscore: np.ndarray,
        evaluation_policy_pscore: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        slate_id: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slates (i.e., ranking or list of actions)

        reward: array-like, shape (<= n_rounds * len_list,)
            Slot-level rewards, i.e., :math:`r_{i}(l)`.

        position: array-like, shape (<= n_rounds * len_list,)
            Indices to differentiate slots/positions in a slate/ranking.

        pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of behavior policy choosing a slate action, i.e., :math:`\\pi_b(a_i|x_i)`.
            This parameter must be unique in each slate.

        evaluation_policy_pscore: array-like, shape (<= n_rounds * len_list,)
            Joint probabilities of evaluation policy choosing a slate action, i.e., :math:`\\pi_e(a_i|x_i)`.
            This parameter must be unique in each slate.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_sips_inputs(
            slate_id=slate_id,
            reward=reward,
            position=position,
            pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        estimated_rewards = self._estimate_round_rewards(
            reward=reward,
            position=position,
            behavior_policy_pscore=pscore,
            evaluation_policy_pscore=evaluation_policy_pscore,
        )
        return self._estimate_slate_confidence_interval_by_bootstrap(
            slate_id=slate_id,
            estimated_rewards=estimated_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )