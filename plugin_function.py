import os
import subprocess
from subprocess import Popen, PIPE
import time
def train_agent(state):
    with open('bluesky_pid.txt','r') as f:
        pid = f.read()
    if state == 0:
    #train the agent
        alter("settings.cfg","case_study_notrain","case_study_b")
        alter("settings.cfg","case_study_test","case_study_b")
        p = Popen('python3 BlueSky.py --sim --detached --scenfile multi_agent.scn',shell=True)
        while True:
            if p.poll() == None:
                time.sleep(1)
            else:
                print('training done')
                show_train_result()
                break
    elif state == 1:
    #continue training
        if pid == '0':
            return
        else:
            run(pid)
    elif state == 2:
    #pause training
        if pid == '0':
            return
        else:
            stop(pid)
    elif state == 3:
    # stop training
        if pid == '0':
            return
        else:
            kill(pid)
        
def show_train_result():
    alter("settings.cfg","case_study_notrain","case_study_b")
    alter("settings.cfg","case_study_test","case_study_b")
    p = Popen('python3 BlueSky_pygame.py',shell=True)
    
def test_before_train():
    alter("settings.cfg","case_study_test","case_study_notrain")
    alter("settings.cfg","case_study_b","case_study_notrain")
    p = Popen('python3 BlueSky_pygame.py',shell=True)
    
def test_after_train():
    alter("settings.cfg","case_study_b","case_study_test")
    alter("settings.cfg","case_study_notrain","case_study_test")
    p = Popen('python3 BlueSky_pygame.py',shell=True)

def killbluesky():
    #when exit a sub interface, use it
    f1 = open(file='bluesky_pid.txt',mode='r')
    pid = f1.read()
    f1.close()
    kill(pid = pid)
    with open('bluesky_pid.txt','w') as f:
        f.write(str(0))
        
def alter(file, old_str,new_str):
    file_data = ""
    with open(file, "r", encoding = "utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)
    
def kill(pid):
    if os.name =='nt':
        cmd = 'taskkill /pid '+ str(pid) + ' /f'
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    elif os.name == 'posix':
        cmd = 'kill '+str(pid)
        try:
            os.system(cmd)
            print(pid, 'killed')
        except Exception as e:
            print(e)
    else:
        print('Undefined os.name')

def stop(pid):
    if os.name == 'posix':
        cmd = 'kill -STOP'+str(pid)
        try:
            os.system(cmd)
            print(pid,'continue')
        except Exception as e:
            print(e)

def run(pid):
    if os.name == 'posix':
        cmd = 'kill -CONT'+str(pid)
        try:
            os.system(cmd)
            print(pid,'continue')
        except Exception as e:
            print(e)
train_agent(0)
