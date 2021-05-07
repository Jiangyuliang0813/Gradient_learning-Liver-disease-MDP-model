import utils
import dataloader
import pickle


def cal_reward(age_init,sick_time,T,p,theta,biospy_cost=0.06,p_cc_d=0.051):
    """
    定义了无需条件判断的奖励计算函数，针对某一特征个体的平均水平
    :return:
    """
    mrp_list = utils.get_mrp_list(age_init,sick_time,T)
    # print('mrp_list for the patient sick in {} is {}'.format(sick_time,mrp_list))
    reward_T = utils.cal_reward_notreat(age_init,sick_time,T)
    mrp = 0
    alpha = 1
    beta = 1
    cal_death = utils.cal_transition_probability
    for i in range(T):
        age = age_init + i
        if i < sick_time:
            p_use = cal_1st_exam(p,theta,i)
            p_ = alpha * p_use
            p_nc_d = cal_death(age)
            mrp = mrp - p_ * biospy_cost + alpha
            alpha = (1-p_nc_d) * alpha
            beta = alpha

        else:
            p_nc_d = cal_death(age)
            p_use = cal_1st_exam(p,theta,i)
            p_ = beta * p_use
            mrp = mrp + p_ * (mrp_list[i] - biospy_cost) + beta
            # print(mrp_list[i])
            beta = (1-p_use)*beta*(1-p_cc_d) * (1 - p_nc_d)


        # print('sick_time:{},detection_cyc:{},time:{},reward_add:{}'.format(self.sick_time,self.T,i,mrp))

    mrp += reward_T * beta

    return mrp


def cal_1st_exam(p,theta,i):
    if isinstance(theta, list):
        if p > theta[i]:
            # print('111111')
            return p
        else:
            # print('222222')
            return 0
    else:
        # print('-------')
        if p > theta:
            # print("333333")
            return p
        else:
            # print("444444444")
            return 0



if __name__ == '__main__':
    feature = dataloader.features_arr
    # print(feature)
    load_model = pickle.load(open("logistic_reg_model.sav",'rb'))
    p = load_model.predict_proba(feature)

    sick_prob = p[2][1]
    # print(sick_prob<0.5)
    for theta in [x/1000 for x in range(200,800,50)]:
        print(cal_1st_exam(sick_prob,theta,0))
        # print(theta)
    # age_init = 80
    # # sick_time = [x for x in range(21)]
    # T = 15
    # sick_time = 0

    # theta_list = [x/1000 for x in range(200,800,50)]
    # for theta in theta_list:
    #     A = cal_reward(age_init,sick_time,T,sick_prob,theta,biospy_cost=0.06,p_cc_d=0.051)
    #     print(A)
    # print(sick_time)
    # print(feature)

    # A = cal_reward(50,1,15,sick_prob,0.2)
    # print(A)