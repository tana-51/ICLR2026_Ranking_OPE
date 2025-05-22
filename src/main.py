from omegaconf import DictConfig, OmegaConf
import hydra
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import seaborn as sns
from sklearn.neural_network import MLPRegressor

import obp
from obp.dataset import(
    linear_reward_function,
    logistic_reward_function,
    linear_behavior_policy,
)

from obp.ope import(
    # SlateOffPolicyEvaluation,
    RegressionModel,
    SlateStandardIPS as IPS,
    SlateIndependentIPS as IIPS,
    SlateRewardInteractionIPS as RIPS,
)

from dataset import SyntheticSlateBanditDataset
from dataset import linear_behavior_policy_logit
from estimator import ClickBasedIPS as CIPS
from ope import OffPolicyEvaluation
from plot import plot

@hydra.main(config_path="../conf",config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.setting.random_state)
    num_runs = cfg.setting.num_runs
    num_data_list = cfg.setting.num_data_list

    dataset = SyntheticSlateBanditDataset(
        n_unique_action=cfg.setting.n_unique_action,
        len_list=cfg.setting.len_list,
        dim_context=cfg.setting.dim_context,
        reward_type=cfg.setting.reward_type,
        reward_structure=cfg.setting.reward_structure,
        decay_function=cfg.setting.decay_function,
        base_reward_function=logistic_reward_function,
        base_reward_function_conversion=linear_reward_function,
        behavior_policy_function=linear_behavior_policy_logit,
        is_factorizable=cfg.setting.is_factorizable,
        random_state=cfg.setting.random_state,
        reward_type_conversion=cfg.setting.reward_type_conversion,
        reward_structure_conversion=cfg.setting.reward_structure_conversion,
        deterministic_user_ratio=cfg.setting.deterministic_user_ratio,
        effect_from_ranking=cfg.setting.effect_from_ranking,
    )

    #evaluation policy
    n_test = cfg.setting.n_test
    context = np.random.normal(size=(n_test, cfg.setting.dim_context))
    evaluation_policy_logit = linear_reward_function(
        context=context,
        action_context=np.eye(cfg.setting.n_unique_action, dtype=int),
        random_state=cfg.setting.random_state,
    )
    pi_e_value = dataset.calc_ground_truth_policy_value_epsilon_greedy(
        context=context,
        evaluation_policy_logit_=evaluation_policy_logit,
        eps=cfg.setting.eps,
    )
    print("pi_e_value", pi_e_value)

    result_df_list = []
    for num_data in num_data_list:
        estimated_policy_value_list = []
        for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
            validation_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=num_data,
                # clip_logit_value=700.0,
            )
            # print("expected_reward_factual", validation_bandit_data["expected_reward_factual"])
            # print("expected_reward_factual_click", validation_bandit_data["expected_reward_factual_click"])
            # print("expected_reward_factual_conversion", validation_bandit_data["expected_reward_factual_conversion"])

            evaluation_policy_logit = linear_behavior_policy_logit(
                context=validation_bandit_data["context"],
                action_context=validation_bandit_data["action_context"],
                random_state=cfg.setting.random_state,
            )

            (
                evaluation_policy_pscore, 
                evaluation_policy_pscore_item_position, 
                evaluation_policy_pscore_cascade,
                evaluation_policy_p_click, 
            )  = dataset.obtain_pscore_given_evaluation_policy_logit_epsilon_greedy(
                context=validation_bandit_data["context"],
                action=validation_bandit_data["action"],
                evaluation_policy_logit_=evaluation_policy_logit,
                eps=cfg.setting.eps,
            )

            ope = OffPolicyEvaluation(
                bandit_feedback=validation_bandit_data,
                ope_estimators=[
                        IPS(estimator_name="IPS", len_list=cfg.setting.len_list), 
                        IIPS(estimator_name="IIPS", len_list=cfg.setting.len_list),  
                        RIPS(estimator_name="RIPS", len_list=cfg.setting.len_list),
                        CIPS(estimator_name="CIPS", len_list=cfg.setting.len_list)
                    ]
            )

            estimated_policy_values = ope.estimate_policy_values(
                evaluation_policy_pscore=evaluation_policy_pscore,
                evaluation_policy_pscore_item_position=evaluation_policy_pscore_item_position,
                evaluation_policy_pscore_cascade=evaluation_policy_pscore_cascade,
                evaluation_policy_p_click=evaluation_policy_p_click,
                behavior_policy_p_click=validation_bandit_data["p_click_factual"],
            )
            estimated_policy_value_list.append(estimated_policy_values)
        
        #summarize result
        result_df = (
            DataFrame(DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        result_df["num_data"] = num_data
        result_df["se"] = (result_df.value - pi_e_value) ** 2
        result_df["bias"] = 0.0
        result_df["variance"] = 0.0

        sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = result_df.loc[result_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            result_df.loc[result_df["est"] == est_, "bias"] = (
                pi_e_value - mean_estimates
            ) ** 2
            result_df.loc[result_df["est"] == est_, "variance"] = (
                estimates - mean_estimates
            ) ** 2
        result_df_list.append(result_df)
        print("max_iw", (evaluation_policy_pscore/ validation_bandit_data["pscore"]).max())
        tqdm.write("=====" * 15)
    
    result_df = pd.concat(result_df_list).reset_index(level=0)

    plot(vary_list=num_data_list, result_df=result_df, variable_name="num_data")

if __name__ == "__main__":
    main()
