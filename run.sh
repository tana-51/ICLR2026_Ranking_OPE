#num_data
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit" 
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="-inf"
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function" setting.n_unique_action=5
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"

#deterministic_ratio
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit"
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="-inf"
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="binary" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function"
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"


#effect_conversion
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit"
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="-inf"
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function" setting.n_unique_action=5
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"

#len_list
# python ./src/main_len_list.py setting.run_file="main_len_list" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_len_list.py setting.run_file="main_len_list" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="-inf"

#estimation_noise
python ./src/main_estimation_noise.py setting.run_file="estimation_noise_test" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"


#kuairec
# python ./real/kuairec.py setting.run_file="kuairec" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" 


# python ./src/main_num_data.py setting.run_file="true_q_r" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf" setting.effect_from_ranking=0.0 setting.num_runs=50

# python ./src/main_estimation_noise.py setting.run_file="main_estimation_noise" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf" setting.effect_from_ranking=0.0 setting.num_runs=50

# python ./src/main_effect_conversion.py setting.run_file="test" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf" setting.effect_from_ranking=0.0 setting.num_runs=50

# python ./src/main_num_data.py setting.run_file="main_num_data_test" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="-inf" setting.effect_from_ranking=1.0 setting.num_runs=50


#最終的な実験
# python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_len_list.py setting.run_file="main_len_list" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"
# python ./src/main_estimation_noise.py setting.run_file="main_estimation_noise" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit" setting.deterministic_user_threshold="inf"

