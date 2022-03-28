""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
import numpy as np
import os
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, settings, navdb, traf, sim, scr, tools
from bluesky import navdb
from bluesky.tools.aero import ft
from bluesky.tools import geo, areafilter
from Multi_Agent.PPO import PPO_Agent
import geopy.distance
import tensorflow.compat.v1 as tf
import random
import pandas as pd
from operator import itemgetter
from visdom import Visdom
from shapely.geometry import LineString
import numba as nb
import time
import csv
tf.disable_eager_execution()

## For running on GPU
# from keras.backend.tensorflow_backend import set_session
# from shapely.geometry import LineString
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#
# sess = tf.Session(config=config)
# set_session(sess)




### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():


    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global agent
    global best_reward
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start
    global wind
    global wind2
    global opts1
    global opts2
    global worst_goals
    global speedlist
    global speedlistpointer
    global routequeuelist
    global routequeuepointer

    routequeuelist = []
    speedlist = []
    num_success_train = []
    num_collisions_train = []

    pid = os.getpid()
    print('pid={}'.format(pid))
    with open('bluesky_pid.txt','w') as f:
        f.write(str(pid))
    
    num_success = []
    num_collisions = []
    previous_action = {}
    last_observation = {}
    observation = {}
    collisions = 0
    success = 0
    num_intruders = 4

    num_ac = 0
    max_ac = 30
    best_reward = -10000000
    ac_counter = 0
    n_states = 5
    route_keeper = np.zeros(max_ac,dtype=int)

    positions = np.load('./routes/case_study_b_route.npy')
    choices = [20,25,30] # 4 minutes, 5 minutes, 6 minutes
    route_queue = random.choices(choices,k=positions.shape[0])
    print(route_queue)
    route_queue = [30,30,20]
    path = "speedbeforetrain.csv"
    f = csv.reader(open(path,'r'))
    for i in f:
        speedlist.append(i)
    speedlistpointer = 0
    path = "route (copy).csv"
    f = csv.reader(open(path,'r'))
    for i in f:
        routequeuelist.append(i)
    routequeuepointer = 0
    
    agent = PPO_Agent(n_states,3,positions.shape[0],100000,positions,num_intruders)
    agent.load("train_model_B.h5")
    counter = 0
    start = time.time()
    opts1 = {
        "title": 'chart example1',
        "xlabel": 'x',
        "ylabel": 'y',
        "width": 300,
        "height": 200,
        "legend": ['goals_made','collisions_made']
    }
    opts2 = {
        "title": 'train_data2',
        "rownames":['KL0','KL1','KL2','KL3','KL4','KL5','KL6','KL7','KL8','KL9','KL10','KL11','KL12','KL13','KL14','KL15','KL16','KL17','KL18','KL19','KL20','KL21','KL22','KL23','KL24','KL25','KL26','KL27','KL28','KL29'],
        "xlabel": 'id',
        "ylabel": 'speed',
        "width": 600,
        "height": 400,
        "numbins":30
    }
    wind = Visdom()
    wind2 = Visdom()
        # 初始化窗口信息
    
    wind.line(X=[0.], # Y的第一个点的坐标
		  Y=[[0.,0.]], # X的第一个点的坐标
		  win = 'train_data', # 窗口的名称
		  opts = opts1 # 图像的标例
)
    wind2.bar(X=np.zeros(30),win = 'train_data2',opts=opts2)

    # Addtional initilisation code
    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     'CASE_STUDY_NOTRAIN',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 12.0,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.

        'update':      update}

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        #'reset':         reset
        #}

    stackfunctions = {
        }



    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin



def update():
    """given a current state in the simulation, allow the agent to select an action.
     "" Then send the action to the bluesky command line ""
    """
    global num_ac
    global counter
    global ac
    global max_ac
    global positions
    global agent
    global success
    global collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global choices
    global start
    global speedlist
    global speedlistpointer
    global routequeuelist
    global routequeuepointer
    global opts2
    global wind2
    global speedvisdomlist
    
    speedvisdomlist = [0]*30

    store_terminal = {}


    if ac_counter < max_ac:  ## maybe spawn a/c based on time, not based on this update interval

        if ac_counter == 0:
            for i in range(len(positions)):
                lat,lon,glat,glon,h = positions[i]
                stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 300'.format(ac_counter,lat,lon,h))
                speedvisdomlist[ac_counter] = 300
                wind2.bar(X=speedvisdomlist,win = 'train_data2',opts=opts2)
                stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter,glat,glon))
                route_keeper[ac_counter] = i
                num_ac += 1
                ac_counter += 1

        else:
            for k in range(len(route_queue)):
                if counter == route_queue[k]:
                    lat,lon,glat,glon,h = positions[k]
                    stack.stack('CRE KL{}, A320, {}, {}, {}, 25000, 300'.format(ac_counter,lat,lon,h))
                    speedvisdomlist[ac_counter] = 300
                    wind2.bar(X=speedvisdomlist,win = 'train_data2',opts=opts2)
                    stack.stack('ADDWPT KL{} {}, {}'.format(ac_counter,glat,glon))
                    route_keeper[ac_counter] = k

                    num_ac += 1
                    ac_counter += 1

                    #route_queue[k] = counter + random.choices(choices,k=1)[0]
                    route_queue[k] = int(routequeuelist[routequeuepointer][1])
                    routequeuepointer = routequeuepointer+1
                    

                    if ac_counter == max_ac:
                        break


    store_terminal = np.zeros(len(traf.id),dtype=int)
    for i in range(len(traf.id)):
        T,type_,nearest = agent.update(traf,i,route_keeper)
        id_ = traf.id[i]

        if T:
            stack.stack('DEL {}'.format(id_))
            speedvisdomlist[int(id_[2:])] = 0
            wind2.bar(X=speedvisdomlist,win = 'train_data2',opts=opts2)
            num_ac -=1
            if type_ == 1:
                collisions += 1
                if collisions%2 ==0:
                    stack.stack('echo plane {} and plane {} had collision'.format(traf.id[nearest],id_))
            if type_ == 2:
                success += 1

            store_terminal[i] = 1

            agent.store(last_observation[id_],previous_action[id_],[np.zeros(last_observation[id_][0].shape),np.zeros(last_observation[id_][1].shape)],traf,id_,route_keeper,type_)

            del last_observation[id_]



    if ac_counter == max_ac and num_ac == 0:
        reset()
        return

    if num_ac == 0 and ac_counter != max_ac:
        return


    if not len(traf.id) == 0:
        ids = []
        new_actions = {}
        n_ac = len(traf.id)
        state = np.zeros((n_ac,5))

        id_sub = np.array(traf.id)[store_terminal != 1]
        ind = np.array([int(x[2:]) for x in traf.id])
        route = route_keeper[ind]

        state[:,0] = traf.lat
        state[:,1] = traf.lon
        state[:,2] = traf.tas
        state[:,3] = route
        state[:,4] = traf.ax

        norm_state,norm_context = getClosestAC(state,traf,route_keeper,previous_action,n_states,store_terminal,agent,last_observation,observation)

        # if norm_state.shape[0] == 0:
        #     import ipdb; ipdb.set_trace()

        policy = agent.act(norm_state,norm_context)

        for j in range(len(id_sub)):
            id_ = id_sub[j]
            #print("id:{}".format(id_))

            # This is for updating s, sp, ...
            if not id_ in last_observation.keys():
                last_observation[id_] = [norm_state[j],norm_context[j]]

            if not id_ in observation.keys() and id_ in previous_action.keys():
                observation[id_] = [norm_state[j],norm_context[j]]

                agent.store(last_observation[id_],previous_action[id_],observation[id_],traf,id_,route_keeper)
                last_observation[id_] = observation[id_]

                del observation[id_]




            #action = np.random.choice(agent.action_size,1,p=policy[j].flatten())[0]
            action = int(speedlist[speedlistpointer][1])
            speedlistpointer = speedlistpointer+1
            speed = agent.speeds[action]
            index = traf.id2idx(id_)
            if action == 1: #hold
                speed = int(np.round((traf.cas[index]/tools.geo.nm)*3600))
                #print(speed)

            stack.stack('SPD {} {}'.format(id_,speed))
            speedvisdomlist[int(id_[2:])] = speed
            wind2.bar(X=speedvisdomlist,win = 'train_data2',opts=opts2)
            new_actions[id_] = action



        previous_action = new_actions

    counter += 1





def reset():
    global best_reward
    global counter
    global num_ac
    global num_success
    global success
    global collisions
    global num_collisions
    global ac_counter
    global route_queue
    global n_states
    global route_keeper
    global previous_action
    global last_observation
    global observation
    global num_success_train
    global num_collisions_train
    global choices
    global positions
    global start
    #global worst_goals
    global speedlistpointer
    global routequeuepointer

    #if (agent.episode_count+1) % 5 == 0:
        #agent.train()
    speedlistpointer = 0
    routequeuepointer = 0
    end = time.time()

    print(end-start)
    goals_made = success

    num_success_train.append(success)
    num_collisions_train.append(collisions)


    success = 0
    collisions = 0


    counter = 0
    num_ac = 0
    ac_counter = 0

    route_queue = random.choices([20,25,30],k=positions.shape[0])
    #print(route_queue)
    route_queue = [30,30,20]


    previous_action = {}
    route_keeper = np.zeros(max_ac,dtype=int)
    last_observation = {}
    observation = {}

    t_success = np.array(num_success_train)
    t_coll = np.array(num_collisions_train)
    np.save('success_train_TEST.npy',t_success)
    np.save('collisions_train_TEST.npy',t_coll)



    if agent.episode_count > 150:
        df = pd.DataFrame(t_success)
        if float(df.rolling(150,150).mean().max()) >= best_reward:
            #agent.save(True,case_study='TEST')
            best_reward = float(df.rolling(150,150).mean().max())


    #agent.save(case_study='TEST')

    #if agent.episode_count ==0:
        #agent.save_worst(goals_made,case_study='TEST')
        #worst_goals = goals_made
    #elif goals_made < worst_goals:
        #agent.save_worst(goals_made,case_study='TEST')
        #worst_goals = goals_made

    print("Episode: {} | Reward: {} | Best Reward: {}".format(agent.episode_count,goals_made,best_reward))

    wind.line(X =[agent.episode_count],Y=[[goals_made,max_ac-goals_made]],win = 'train_data',update='append',opts = opts1)

    agent.episode_count += 1

    if agent.episode_count == agent.numEpisodes:
        stack.stack('STOP')

    stack.stack('IC multi_agent.scn')

    start = time.time()


def getClosestAC(state,traf,route_keeper,new_action,n_states,store_terminal,agent,last_observation,observation):
    n_ac = traf.lat.shape[0]
    norm_state = np.zeros((len(store_terminal[store_terminal!=1]),5))

    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1,1)

    d = geo.latlondist_matrix(np.repeat(state[:,0],n_ac),np.repeat(state[:,1],n_ac),np.tile(state[:,0],n_ac),np.tile(state[:,1],n_ac)).reshape(n_ac,n_ac)
    argsort = np.array(np.argsort(d,axis=1))


    total_closest_states = []
    route_count = 0
    i = 0
    j = 0

    max_agents = 1

    count = 0
    for i in range(d.shape[0]):
        r = int(state[i][3])
        lat,lon,glat,glon,h = agent.positions[r]
        if store_terminal[i] == 1:
            continue
        ownship_obj = LineString([[state[i][1],state[i][0],31000],[glon,glat,31000]])

        norm_state[count,:] = agent.normalize_that(state[i],'state',id_=traf.id[i])
        closest_states = []
        count += 1

        route_count = 0

        intruder_count = 0

        for j in range(len(argsort[i])):


            index = int(argsort[i][j])

            if i == index:
                continue

            if store_terminal[index] == 1:
                continue

            route = int(state[index][3])


            if route == r and route_count == 2:
                continue


            if route == r:
                route_count += 1

            lat,lon,glat,glon,h = agent.positions[route]
            int_obj = LineString([[state[index,1],state[index,0],31000],[glon,glat,31000]])

            if not ownship_obj.intersects(int_obj):
                continue


            if not route in agent.intersection_distances[r].keys() and route != r:
                continue


            if d[i,index] > 100:
                continue

            max_agents = max(max_agents,j)


            if len(closest_states) == 0:
                closest_states = np.array([traf.lat[index], traf.lon[index], traf.tas[index],route,traf.ax[index]])
                closest_states = agent.normalize_that(norm_state[count-1],'context',closest_states,state[i],id_=traf.id[index])
            else:
                adding = np.array([traf.lat[index], traf.lon[index], traf.tas[index],route,traf.ax[index]])
                adding = agent.normalize_that(norm_state[count-1],'context',adding,state[i],id_=traf.id[index])

                closest_states = np.append(closest_states,adding,axis=1)

            intruder_count += 1

            if intruder_count == agent.num_intruders:
                break



        if len(closest_states) == 0:
            closest_states = np.array([0,0,0,0,0,0,0]).reshape(1,1,7)


        if len(total_closest_states) == 0:
            total_closest_states = closest_states
        else:

            total_closest_states = np.append(tf.keras.preprocessing.sequence.pad_sequences(total_closest_states,agent.num_intruders,dtype='float32'),tf.keras.preprocessing.sequence.pad_sequences(closest_states,agent.num_intruders,dtype='float32'),axis=0)



    if len(total_closest_states) == 0:
        total_closest_states = np.array([0,0,0,0,0,0,0]).reshape(1,agent.num_intruders,7)


    return norm_state,total_closest_states
