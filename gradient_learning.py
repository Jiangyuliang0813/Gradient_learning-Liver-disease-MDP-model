import main_theta
import dataloader
import warnings
import json
import pickle
import argparse


parser = argparse.ArgumentParser(description='Parmarmeter')
parser.add_argument('--theta',type=float)
parser.add_argument('--age',type=int)

args = parser.parse_args()

warnings.filterwarnings('ignore')

def cal_j_theta(theta_T,length,p_features,age,sick_time_list,weight,T):
    reward_theta = main_theta.cal_avg_mrp_one(theta_T,length,p_features,age,sick_time_list,weight,T)
    return reward_theta

def train(theta_train_data,age,sick_time_list,T=15):
    lr = 0.05
    epsilon = 0.001
    epoch = 50
    reward_process = []
    theta_process = []
    weight = main_theta.sick_time_weight(sick_time_list)
    features = dataloader.features_arr
    length = features.shape[0]
    loaded_model = pickle.load(open('logistic_reg_model.sav','rb'))
    p_features = loaded_model.predict_proba(features)
    append_reward = reward_process.append
    append_theta = theta_process.append

    for i in range(epoch):
        gradient = []
        j_theta = cal_j_theta(theta_train_data,length,p_features,age,sick_time_list,weight,T)
        append_reward(j_theta)
        print("Total Reward is {}".format(j_theta))
        append_gradient = gradient.append
        for k in range(T):
            print('The dimension k is {}'.format(k))
            theta_T_epsilon = theta_train_data
            theta_T_epsilon[k] += epsilon
            j_theta_epsilon = cal_j_theta(theta_T_epsilon,length,p_features,age,sick_time_list,weight,T)
            gradient_k = (j_theta_epsilon - j_theta) / epsilon
            append_gradient(gradient_k)

        theta_train_data = [x + lr*y for x, y in zip(theta_train_data,gradient)]

        append_theta(theta_train_data)

        print("epoch:{}\ngradient:{}\ntheta:{}".format(i,gradient,theta_train_data))

    return theta_train_data,reward_process,theta_process


if __name__ == '__main__':
    T = 15
    theta_T = []


    age = args.age
    str1 = str(age)
    file_name = str1 + 'train2_information.json'
    # with open(file_name, 'r') as json_file:
    #     train_information = json.load(json_file)
    #
    # theta_T = train_information["best_theta"]
    for i in range(T):
        theta_T.append(args.theta)

    sick_time_list = range(21)
    theta_T_final, reward_list, theta_list = train(theta_T,age,sick_time_list,T)
    print('the best theta is {}'.format(theta_T_final))
    print('the best reward is {}'.format(reward_list[-1]))

    train_information = {
        "reward": reward_list,
        "theta": theta_list,
        "best_theta": theta_T_final,
        "base_theta":theta_T
    }

    json_str = json.dumps(train_information)
    file_name2 = file_name
    with open(file_name2, 'w') as json_file:
        json_file.write(json_str)
        print('训练及结果已经写入文件train_information.json')