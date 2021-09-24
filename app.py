# LIBRARY
import tkinter
from tkinter import *
import tkinter.messagebox
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import ttk
from datetime import datetime
from time import strftime, gmtime
from tkinter import messagebox as mb
import Model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
pd.set_option('display.max_columns',None)
import nltk
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# function to set parameters of gui
def main_wind():
    global main_window
    main_window = Tk()
    screen_width = main_window.winfo_screenwidth()
    screen_height = main_window.winfo_screenheight()
    main_window.geometry("%dx%d" % (screen_width, screen_height))
    main_window.resizable(0, 0)
    try:
        main_window.wm_iconbitmap('icon.png')
    except Exception as icon_absent:
        tkinter.messagebox.showinfo("Icon Not Found", "icon.png File Missing")
    main_window.title("Detection of Terrorist threats on Twitter")

#function to rise a new frame within the GUI
def raise_frame(frame):
    frame.tkraise()

#function to show time and date below GUI heading
def showtime(cs_framet):
    def time():
        string = strftime("Date: %d-%m-%Y                    Time: %H:%M:%S")
        e.config(text=string)
        e.after(1000, time)
    try:
        e = Label(cs_framet, bg="skyblue", fg="#094c72", width=1000)
        e.config(font=("italian", 12))
        e.pack()
        time()
    except Exception as ext:
        pass

def printInput():
    lbl = Label(main_window,text = "Terrorist Found !!")
    lbl.place(x=400,y=300)

#function to test using svm-demo algorithm
def svmDemo_Click():
    global selectAlgo_Frame, fig
    inputtxt = Text(main_window, height = 5, width = 25)   
    inputtxt.place(x=550,y=350)  
    printButton = Button(main_window, text = "Check", command =lambda: [printInput()]) 
    printButton.place(x=650,y=400)
    model = np.load('terriorist_model.npy', 'rb', allow_pickle="True")
    result = model.predict([inputtxt])
    prob = model.predict_proba([inputtxt])
    print(type(result))
    if(result):    
        lbl = Label(main_window,text = "Terrorist Threat Not Found")
        lbl.place(x=400,y=300)
    else:
        lbl = Label(main_window,text = "Terrorist Threat Found")
        lbl.place(x=400,y=300)

#function to test using lg-demo algorithm
def lgDemo_Click():
    global selectAlgo_Frame
    inputtxt = Text(main_window, height = 5, width = 25)   
    inputtxt.place(x=550,y=350)  
    printButton = Button(main_window, text = "Check", command =lambda: [printInput()]) 
    printButton.place(x=650,y=400)
    

#function to test using nb-demo algorithm
def nbDemo_Click():
    global selectAlgo_Frame
    inputtxt = Text(main_window, height = 5, width = 25)   
    inputtxt.place(x=550,y=350)  
    printButton = Button(main_window, text = "Check", command =lambda: [raise_frame(selectAlgo_Frame),printInput()]) 
    printButton.place(x=650,y=400)

    
    
#function to load main window of GUI
def main_menu():

    load_dataset_frame = Frame(main_window)
    load_dataset_frame.place(x=320, y=250, width=1000, height=517)

    algorithm_check_frame = Frame(main_window)
    algorithm_check_frame.place(x=320, y=250, width=1000, height=517)

    landing_page_frame = Frame(main_window, bg="#ebf9ff")
    landing_page_frame.place(width=1600, height=200)

    accuracy_frame = Frame(main_window)
    accuracy_frame.place(x=600, y=300, width=500, height=517)

    name = Label(landing_page_frame, text="Detection of Terrorist threats on Twitter", height=3, width=80, bg="#026aa7", fg="white")
    name.config(font=("italian", 31))
    name.pack(padx=1, pady=1)

    showtime(landing_page_frame)

    footer = Label(main_window, fg = "#026aa7", text = "ALL RIGHTS RESERVED", bg = "#add8e6",
                        font = ("italian", 12, 'bold'), width = 155, height = 3)
    footer.place(y = 780)

    selectAlgo_Frame = Frame(main_window)
    selectAlgo_Frame.place(x = 320, y = 250, width = 1000, height = 517)

    
    #function to set sizes and properties of all frames
    def win_size(window):
        screen_width = window.winfo_screenwidth() - 8
        screen_height = window.winfo_screenheight() - 40
        window.geometry("%dx%d" % (screen_width, screen_height))
        window.resizable(0, 0)
        try:
            main_window.wm_iconbitmap('icon.png')
        except Exception as icon_absent:
            tkinter.messagebox.showinfo("Icon Not Found", "icon.png File Missing")
        window.title("Detection of Terrorist threats on Twitter")


    #function for btn 4
    def demo():
        label_name6 = Label(selectAlgo_Frame, text="Demo Algorithm", anchor=CENTER, fg="#026aa7",
                           justify="center",
                           font=("italian", 20, 'bold'))
        label_name6.place(x=520)

        demo1_Btn = Button(selectAlgo_Frame, text = "SVM Demo",pady=2, height=3, width=30, foreground="white",bg="#298fca", activeforeground="skyblue", activebackground="#026aa7",font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame),svmDemo_Click()])
        demo1_Btn.place(x= 450, y = 120)

        demo2_Btn = Button(selectAlgo_Frame, text = "Logistic Regression Demo",pady=2, height=3, width=30, foreground="white",bg="#298fca", activeforeground="skyblue", activebackground="#026aa7",font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame), lgDemo_Click()])
        demo2_Btn.place(x= 450, y = 240)

        demo3_Btn = Button(selectAlgo_Frame, text = "Naive Bayes Demo",pady=2, height=3, width=30, foreground="white",bg="#298fca", activeforeground="skyblue", activebackground="#026aa7",font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame), nbDemo_Click()])
        demo3_Btn.place(x= 450, y = 360)


    #function to load svm sav model
    def load_model_svm():
        model = pickle.load(open('terriorist_model.sav','rb'))
        lbl = Label(main_window,text = "Model Trained")
        lbl.place(x=400,y=300)

    #function to load naive bayes sav model
    def load_model_nb():
        model = pickle.load(open('bayes_model.sav','rb'))
        lbl = Label(main_window,text = "Model Trained")
        lbl.place(x=400,y=300)


    #function to load logistic regression sav model
    def load_model_lg():
        model = pickle.load(open('logistic_model.sav','rb'))
        lbl = Label(main_window,text = "Model Trained")
        lbl.place(x=400,y=300)
              
    #Function for btn 3
    def select_algo_btn():
        label_name3 = Label(selectAlgo_Frame, text="Select Algorithm", anchor=CENTER, fg="#026aa7",justify="center",font=("italian", 20, 'bold'))
        label_name3.place(x=520)
    
        Algo1 = Button(selectAlgo_Frame, text="SVM", pady=2, height=3, width=30, foreground="white",
                                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7",
                                  font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame),load_model_svm()])
        Algo1.place(x=450, y=120)
        Algo2 = Button(selectAlgo_Frame, text="Logistic Regression", pady=2, height=3, width=30, foreground="white",
                                  bg="#298fca", activeforeground="skyblue",
                                  activebackground="#026aa7", font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame),load_model_lg()])
        
        Algo2.place(x=450, y=240)
        Algo3 = Button(selectAlgo_Frame, text="Naive Bayes", pady=2, height=3, width=30, foreground="white",
                                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7",
                                  font=("italian", 13), command=lambda: [raise_frame(selectAlgo_Frame),load_model_nb()])
        Algo3.place(x=450, y=360)


    #fuction to show accuracy graph on btn 2
    def graph():
        global fig
        fig = Figure(figsize = (5,5), dpi = 100)
        plot1 = fig.add_subplot(111)
        plot1.plot(Model.acc)
        canvas = FigureCanvasTkAgg(fig, master = main_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas,main_window)
        toolbar.update()
        canvas.get_tk_widget().pack()


    #function for btn 1
    def load_data_frame():
        try:
            terroriest = pd.read_csv('twitterdataset.csv',header=None) 
            normal = pd.read_csv('normal.csv',encoding='ISO-8859-1',header=None)

            terroriest.columns = ['target','date','text'] 
            terroriest = terroriest[['target','text']]
            terroriest['target'] = 1
            pd.DataFrame(terroriest['text'])

            normal = normal.iloc[:3000,4:]
            normal.columns = ['target','text']
            normal['target'] = 0 
            df = pd.concat([normal,terroriest],axis=0)
            #display few records
            Output = Text(main_window, height = 20, width = 80)
            Output.place(x=550,y=350)
            Output.insert(END, df.head(15))
            
            label_name5 = Label(load_dataset_frame, text="Data Set Loaded", anchor=CENTER, fg="#026aa7",
                            justify="center",
                            font=("italian", 20, 'bold'))
            label_name5.place(x=520)
            
        except:
            tkinter.messagebox.showinfo("DataSet Not Found", "csv File Missing")


#   MAIN WINDOW BUTTONS
    btn1 = Button(main_window, text="Load Dataset", padx=4, pady=2, height=3, width=28, foreground="white",
                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7", font=("italian", 13, 'bold'),
                  command=lambda: [raise_frame(load_dataset_frame), load_data_frame()])
    btn2 = Button(main_window, text="Check Model Accuracy", padx=4, pady=2, height=3, width=28, foreground="white",
                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7", font=("italian", 13, 'bold'),
                  command=lambda: [raise_frame(algorithm_check_frame), graph()])
    btn3 = Button(main_window, text="Train and Test Model", padx=4, pady=2, height=3, width=28, foreground="white",
                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7", font=("italian", 13, 'bold'),
                  command=lambda: [raise_frame(selectAlgo_Frame), select_algo_btn()])
    btn4 = Button(main_window, text="Sample Demo", padx=4, pady=2, height=3, width=28, foreground="white",
                  bg="#298fca", activeforeground="skyblue", activebackground="#026aa7", font=("italian", 13, 'bold'),
                  command=lambda: [raise_frame(selectAlgo_Frame), demo()])

    btn1.grid(row=5, column=0)
    btn1.place(relx=0, rely=0.2)
    btn2.grid(row=7, column=0)
    btn2.place(relx=0, rely=0.4)
    btn3.grid(row=9, column=0)
    btn3.place(relx=0, rely=0.6)
    btn4.grid(row=11, column=0)
    btn4.place(relx=0, rely=0.8)


if __name__ == '__main__':
    
    main_wind()
    main_menu()
    main_window.mainloop()
