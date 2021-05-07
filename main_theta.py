import patientclass
import dataloader
import pickle

def cal_avg_mrp_one(theta,length,p_features,age,sick_time_list,weight_list,T):
    Amrp = 0
    cal_mrp = patientclass.cal_reward
    # weight_list = sick_time_weight(sick_time_list)
    for i in range(length):
        p = p_features[i][1]
        for k in range(len(sick_time_list)):
            weight = weight_list[k]
            mrp_A = cal_mrp(age,sick_time_list[k],T,p,theta)
            Amrp += mrp_A * weight
            # print("name:{},sick time:{},age:{},mrp:{},total_mrp:{}".format(i,sick_time_list[k],age,mrp_A,Amrp))
    Amrp = Amrp / length

    return Amrp

def sick_time_weight(sick_time_list):
    sick_length = len(sick_time_list)
    p_cc_nc = 0.039
    sick_prob = []
    prob = 1
    prob_sum = 0
    append = sick_prob.append
    for i in range(sick_length):
        prob = (1 - p_cc_nc) * prob
        append(prob)
        prob_sum += prob
    # sick_prob.append(1-prob_sum)
    sick_prob_normalize = [x/prob_sum for x in sick_prob]
    return sick_prob_normalize


# print(sick_time_weight([x for x in range(21)]))
if __name__ == '__main__':
    weight_list = sick_time_weight([x for x in range(21)])
    features = dataloader.features_arr
    length = features.shape[0]
    loaded_model = pickle.load(open('logistic_reg_model.sav', 'rb'))
    p_features = loaded_model.predict_proba(features)
    theta = 0.2
    age = 50
    sick_time_list = [x for x in range(21)]
    T = 15
    mrp = cal_avg_mrp_one(theta, length, p_features, age, sick_time_list, weight_list, T)

    print('----------------------------------------------------------------------')
    print('mrp = {} '.format(mrp))