from tkinter import *
import os
import subprocess
class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
    
    def createWidgets(self):
        self.helloLabel = Label(self, text='Hello, world!')
        self.helloLabel.pack()
        self.quitButton = Button(self,text='Quit',command=killbluesky)
        self.quitButton.pack()
        self.anotherButton = Button(self,text='train',command=self.train)
        self.anotherButton.pack()
        self.otherButton = Button(self,text='other',command=self.show_before_train)
        self.otherButton.pack()
    def train(self):
        alter("settings.cfg","case_study_test","case_study_b")
        alter("settings.cfg","case_study_show","case_study_b")
        subprocess.run('python3 /home/bkz/Autonomous-ATC-N_Closest/BlueSky.py --sim --detached --scenfile multi_agent.scn',shell=True)
        #os.system('python3 BlueSky.py --sim --detached --scenfile multi_agent.scn')
    def show_before_train(self):
        alter("settings.cfg","case_study_b","case_study_test")
        alter("settings.cfg","case_study_show","case_study_test")
        os.system('python3 BlueSky_pygame.py')
    def show_after_train(self):
        alter("settings.cfg","case_study_test","case_study_show")
        alter("settings.cfg","case_study_b","case_study_show")
        os.system('python3 BlueSky_pygame.py')
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
def killbluesky():
    f1 = open(file='bluesky_pid.txt',mode='r')
    pid = f1.read()
    f1.close()
    kill(pid = pid)
app = Application()
app.master.title('Hello World')
app.mainloop()
