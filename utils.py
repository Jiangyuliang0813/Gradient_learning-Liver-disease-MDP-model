

def cal_transition_probability(age,age_list_50_99_5=[0.006,0.009,0.013,0.018,0.027,0.043,0.072,0.122,0.200,0.299]):
    """
    计算病人在某年龄下的转移概率
    :param age: 病人年龄
    :return: 治疗状态到死亡状态的转移概率
    """
    list_num = (age-50)//5
    treat_to_death = age_list_50_99_5[list_num]
    return treat_to_death
# print(cal_transition_probability(76))

def cal_mrp(age_init,t,T,reward_treat=0.95,reward_death=0):
    """
    计算病人当前时间下被治疗后的总mrp
    :param age_init: 病人进入监控的初始年龄
    :param t: 病人被治疗的时间
    :return: 病人在该时间被治疗后直到99岁的总奖励
    """
    assert age_init >= 50, '病患年龄小于50'
    assert t < T, '病患已处于监控最后一天'

    age_now = age_init + t
    mrp = 0

    while(age_now<=99):
        death_prob = cal_transition_probability(age_now)
        live_prob = (1-death_prob) * (1- 0.045)
        mrp += reward_treat * live_prob + death_prob * reward_death
        # print('reward:{},reward_treat:{},live_prob:{},age:{}'.format(mrp, reward_mrp_treat, live_prob, age_now))

        reward_treat = reward_treat * live_prob
        reward_death = reward_death * live_prob
        age_now = age_now + 1

    return mrp
# print(cal_mrp(50,11,15))

def cal_mrp_sick(age_init,t,i,T, reward_treat=0.95, reward_death=0, reward_normal=1):
    """
    计算病人在未接受治疗后的总反馈,sick time 如果为1000则不会患病
    :param age_init: 病人进入监控的初始年龄
    :param t: 病人被开始计算的时间
    :return: 病人未接受治疗后的反馈
    """

    assert age_init >= 50, '病患年龄小于50'
    assert t < T, '病患已处于监控最后一天'

    if i >= 20:
        i = 1000

    age_now = age_init + t
    mrp = 0
    while(age_now<=99):
        if t<i:
            death_prob = cal_transition_probability(age_now)
            live_prob = (1-death_prob)
            mrp += reward_normal * live_prob + death_prob * reward_death
        # print('reward:{},reward_treat:{},live_prob:{},age:{}'.format(mrp, reward_mrp_treat, live_prob, age_now))
        else:
            death_prob = cal_transition_probability(age_now)
            live_prob = (1-death_prob) * (1 - 0.051)
            mrp += reward_normal * live_prob + death_prob * reward_death
        reward_normal = reward_normal * live_prob
        # print(reward_mrp_normal)
        reward_death = reward_death * live_prob
        age_now = age_now + 1
        t += 1

    return mrp

def cal_reward_notreat(age_init,sick_time,T):
    """
    计算监控周期内均没有进行活体检测的奖励
    :param age_init: 初始年龄
    :return: 初始年龄到99岁的反馈
    """
    return cal_mrp_sick(age_init,T-1,sick_time,T)

# print(cal_reward_notreat(50,5,15))


def get_mrp_list(age_init,i,T):
    """
    返回监控周期内的mrp列表 [mrp_i,...,mrp_T-1],问题在于
    :param age_init: 病人初始年龄
    :param i: 病人患病时期
    :return: 病人患病后治疗奖励列表
    """
    assert age_init>=50, '病人未超过50岁'
    mrp_list = []
    t = 0
    append = mrp_list.append
    while(t<T):
        if t < i:
            append(0.00)
        else:
            mrp = cal_mrp(age_init,t,T)
            append(mrp)
        t += 1
    return mrp_list

# print(get_mrp_list(50,0,15))
