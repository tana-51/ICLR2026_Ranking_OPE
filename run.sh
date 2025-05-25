#num_data
python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function"
python ./src/main_num_data.py setting.run_file="main_num_data" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"

#deterministic_ratio
python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function"
python ./src/main_deterministic_ratio.py setting.run_file="main_deterministic_ratio" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"


#deterministic_ratio
python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_behavior_policy_logit"
python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=5 setting.evaluation_policy_logit="linear_reward_function"
python ./src/main_effect_conversion.py setting.run_file="main_effect_conversion" setting.reward_type_conversion="continuous" setting.len_list=6 setting.evaluation_policy_logit="linear_reward_function"
