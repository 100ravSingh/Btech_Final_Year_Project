#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import loadtxt
import xgboost
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import *
import time
from optparse import OptionParser
import matplotlib
from matplotlib import pyplot as plt
from tikzplotlib import save as tikz_save
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot
import re
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.filedialog import asksaveasfile
# import app1 as app
# import csv
import pandas as pd




model = XGBRegressor()

# cross validation
def cv(random_state, poly_degree, model, tester,problem, print_folds=True):
    interaction_only = False
    MEAN = []
    if problem.lower() == "compressive" or problem.lower() == "b_compressive":
        data_file ='compressive_strength.csv'
        data = pd.read_csv(data_file)
        interaction_only = True

    elif problem.lower() == "tensile" or problem.lower() == "b_tensile":
        data_file = 'tensile_strength.csv'
        data = pd.read_csv(data_file)

    elif problem.lower() == "test2":
        data_file ='data2set.csv'
        data = pd.read_csv(data_file)
    else:
        print("The problem has to be compressive or tensile or test2")
        return

    data = data.values
    n_data_cols = np.shape(data)[1]
    n_features = n_data_cols - 1

    # retrieve data for features
    X = np.array(data[:, :n_features])
    y = np.array(data[:, n_features:])
    # split into 10 folds with shuffle
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    start_time = time.time()
    scores = []
    fold_index = 0

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = tester
        #y_test = y[test_index]

        X_scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = y_scaler.fit_transform(y_train)

        if poly_degree >= 1:
            poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            # print ('Total number of features: ', X_train.size)

        model.fit(X_train, y_train.ravel())

        y_pred = model.predict(X_test)
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        MEAN.append(y_pred)
        
    if problem.lower() == "b_compressive" or problem.lower() == "b_tensile":
        return y_pred
    if problem.lower() == "compressive" or problem.lower() == "tensile":
        return np.average(MEAN)
        



def run_xgb(random_state, poly_degree, n_estimators, max_depth, learning_rate,objective, problem,tester):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         objective=objective, random_state=random_state)
    outpts = cv(random_state, poly_degree, model,tester, problem=problem)
    return outpts
    


def calci(testing,natures):
    
    
    if natures == "compressive":
        outpts2 = run_xgb(random_state=0, poly_degree=1, n_estimators=1400, max_depth=6,
        learning_rate=0.15, objective="reg:logistic", problem="compressive",tester=testing)

    if natures == "b_compressive":
        outpts2 = run_xgb(random_state=0, poly_degree=1, n_estimators=1400, max_depth=6,
        learning_rate=0.15, objective="reg:logistic", problem="b_compressive",tester=testing)
    
    if natures == "tensile":
        outpts2 = run_xgb(random_state=0, poly_degree=2, n_estimators=700, max_depth=6,
        learning_rate=0.09, objective="reg:logistic", problem="tensile",tester=testing)
    
    if natures == "b_tensile":
        outpts2 = run_xgb(random_state=0, poly_degree=2, n_estimators=700, max_depth=6,
        learning_rate=0.09, objective="reg:logistic", problem="b_tensile",tester=testing)
        
    return outpts2



# In[12]:


root = Tk()
root.geometry("1000x800")
root.title('HPC Strength Calculator')

label_0 = Label(root,text="VSSUT HPC Strength Calculator",width = 40,font=("bold",20))
label_0.place(x=150,y=30)


Button(root,text = "Compressive_Page",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_mycomp()).place(x=250,y=80)
Button(root,text = "Tensile_Page",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_myten()).place(x=420,y=80)
Button(root,text = "Batch_Compute",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_mybat()).place(x=560,y=80)
Button(root,text = "About us",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_about()).place(x=700,y=80)

def set_about():
    message = "Design and Develop by Sourav Singh and Prof S.K Patro .\nInstitution : VSSUT Burla, Odisha. \nYear: 2020-Present."
    messagebox.showinfo("About Developers", message)
    


def set_mycomp():
    
    def clear_screen():
        label_cement.destroy()
        label_BF.destroy()
        label_ash.destroy()
        label_water.destroy()
        label_plasticizer.destroy()
        label_coarse.destroy()
        label_fine.destroy()
        label_age.destroy()
        label_comp.destroy()
        cement_1.destroy()
        BF_1.destroy()
        fly_ash.destroy()
        water.destroy()
        plasticizer.destroy()
        coarse.destroy()
        fine.destroy()
        day.destroy()
        B1.destroy()
        B2.destroy()
        C1.destroy()
    
    def compression_data():
        Cement = float(cement_1.get())
        BF = float(BF_1.get())
        Fly_Ash = float(fly_ash.get())
        Water = float(water.get())
        Plasticizer = float(plasticizer.get())
        Coarse = float(coarse.get())
        Fine = float(fine.get())
        Day = int(day.get())
    
    
        T1 = [[Cement,BF,Fly_Ash,Water,Plasticizer,Coarse,Fine,Day]]
    
        compressive = calci(T1,"compressive")
    
        label_comp.config(text = " Compressive " + str(compressive) + " MPa")
    
    
    
    def reset_data():
        cement_1.delete(0,END)
        BF_1.delete(0,END)
        fly_ash.delete(0,END)
        water.delete(0,END)
        plasticizer.delete(0,END)
        coarse.delete(0,END)
        fine.delete(0,END)
        day.delete(0,END)
        label_comp.config(text="")
    
    label_cement = Label(root,text="Cement (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_cement.place(x=5,y=130)
    cement_1 = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    cement_1.place(x=600,y=130)
    label_BF = Label(root,text="BF Slag (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_BF.place(x=5,y=170)
    BF_1 = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    BF_1.place(x=600,y=170)
    label_ash = Label(root,text="Fly ash (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_ash.place(x=5,y=210)
    fly_ash = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    fly_ash.place(x=600,y=210)
    label_water = Label(root,text="Water (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_water.place(x=5,y=250)
    water = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    water.place(x=600,y=250)
    label_plasticizer = Label(root,text="Superplasticizer (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_plasticizer.place(x=5,y=290)
    plasticizer = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    plasticizer.place(x=600,y=290)
    label_coarse = Label(root,text="Coarse Aggregrate (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_coarse.place(x=5,y=330)
    coarse = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    coarse.place(x=600,y=330)
    label_fine = Label(root,text="Fine Aggregrate (Kg per m^3 mixture)",width = 40,font=("bold",15))
    label_fine.place(x=5,y=370)
    fine = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    fine.place(x=600,y=370)
    label_age = Label(root,text="Age ( Day )",width = 40,font=("bold",15))
    label_age.place(x=5,y=410)
    day = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    day.place(x=600,y=410)
    label_comp = Label(root,text="",width = 40,font=("bold",15))
    label_comp.place(x=200,y=625)
    
    
    B1=Button(root,text = "Compressive",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:compression_data())
    B1.place(x=300,y=500)
    B2=Button(root,text = "Reset Data",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:reset_data())
    B2.place(x=500,y=500)
    C1=Button(root,text = "Clear Screen",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:clear_screen())
    C1.place(x=700,y=500)
    
        
    

def set_myten():
    
    def clear_tensile():
        
        label_cement_comp.destroy()
        label_cement_ten.destroy()
        label_curing.destroy()
        label_Dmax.destroy()
        label_powder.destroy()
        label_fineness.destroy()
        label_WB.destroy()
        label_WC.destroy()
        label_water.destroy()
        label_sand.destroy()
        label_slump.destroy()
        label_concrete.destroy()
        label_tens.destroy()
        cement_1_comp.destroy()
        cement_ten.destroy()
        curing_age.destroy()
        Dmax.destroy()
        powder.destroy()
        fineness.destroy()
        WB.destroy()
        WC.destroy()
        water.destroy()
        sand.destroy()
        slump.destroy()
        concrete.destroy()
        B3.destroy()
        B4.destroy()
        C2.destroy()
    
    def tensile_data():
        Cement_comp = float(cement_1_comp.get())
        Cement_ten=float(cement_ten.get())
        Curing_Age=int(curing_age.get())
        Dmax_val=float(Dmax.get())
        Powder=float(powder.get())
        Fineness=float(fineness.get())
        WB_val=float(WB.get())
        WC_val=float(WC.get())
        Water=float(water.get())
        Sand=float(sand.get())
        Slump=float(slump.get())
        Concrete=float(concrete.get())
    
    
        T1 = [[Cement_comp,Cement_ten,Curing_Age,Dmax_val,Powder,Fineness,WB_val,WC_val,Water,Sand,Slump,Concrete]]
    
        tensile = calci(T1,"tensile")
    
        label_tens.config(text = " Split Tensile : " + str(tensile) + " MPa")
    
    def reset_tensile():
        cement_1_comp.delete(0,END)
        cement_ten.delete(0,END)
        curing_age.delete(0,END)
        Dmax.delete(0,END)
        powder.delete(0,END)
        fineness.delete(0,END)
        WB.delete(0,END)
        WC.delete(0,END)
        water.delete(0,END)
        sand.delete(0,END)
        slump.delete(0,END)
        concrete.delete(0,END)
        label_tens.config(text="")
    
    label_cement_comp = Label(root,text="Compressive Strengh of Cement (MPa)",width = 40,font=("bold",15))
    label_cement_comp.place(x=5,y=130)
    cement_1_comp = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    cement_1_comp.place(x=600,y=130)
    label_cement_ten = Label(root,text="Tensile Strengh of Cement (MPa)",width = 40,font=("bold",15))
    label_cement_ten.place(x=5,y=170)
    cement_ten = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    cement_ten.place(x=600,y=170)
    label_curing = Label(root,text="Curing Age (Day)",width = 40,font=("bold",15))
    label_curing.place(x=5,y=210)
    curing_age = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    curing_age.place(x=600,y=210)
    label_Dmax = Label(root,text="Dmax of crushed stone (mm)",width = 40,font=("bold",15))
    label_Dmax.place(x=5,y=250)
    Dmax = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    Dmax.place(x=600,y=250)
    label_powder = Label(root,text="Stone powder content in sand (%)",width = 40,font=("bold",15))
    label_powder.place(x=5,y=290)
    powder = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    powder.place(x=600,y=290)
    label_fineness = Label(root,text="Fineness modulus of Sand",width = 40,font=("bold",15))
    label_fineness.place(x=5,y=330)
    fineness = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    fineness.place(x=600,y=330)
    label_WB = Label(root,text="W/B ",width = 40,font=("bold",15))
    label_WB.place(x=5,y=370)
    WB = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    WB.place(x=600,y=370)
    label_WC = Label(root,text="Water to cement ratio",width = 40,font=("bold",15))
    label_WC.place(x=5,y=410)
    WC = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    WC.place(x=600,y=410)
    label_water = Label(root,text="Water (Kg/m^3)",width = 40,font=("bold",15))
    label_water.place(x=5,y=450)
    water = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    water.place(x=600,y=450)
    label_sand = Label(root,text="Sand ratio",width = 40,font=("bold",15))
    label_sand.place(x=5,y=490)
    sand = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    sand.place(x=600,y=490)
    label_slump = Label(root,text="Slump (mm)",width = 40,font=("bold",15))
    label_slump.place(x=5,y=530)
    slump = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    slump.place(x=600,y=530)
    label_concrete = Label(root,text="Compressive strength of concrete (MPa)",width = 40,font=("bold",15))
    label_concrete.place(x=5,y=570)
    concrete = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
    concrete.place(x=600,y=570)
    
    label_tens = Label(root,text="",width = 40,font=("bold",15))
    label_tens.place(x=100,y=655)
    
    
    B3=Button(root,text = "Tensile",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:tensile_data())
    B3.place(x=550,y=620)
    B4=Button(root,text = "Reset Data",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:reset_tensile())
    B4.place(x=665,y=620)
    C2=Button(root,text = "Clear Screen",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:clear_tensile())
    C2.place(x=820,y=620)
    
    

def set_mybat():
    
    def clear_batch():
        label_file.destroy()
        label_cement_batch.destroy()
        label_cement_batch2.destroy()
        label_cement_batch3.destroy()
        label_cement_batch4.destroy()
        label_cement_batch5.destroy()
        B5.destroy()
        B6.destroy()
        B7.destroy()
        C3.destroy()
        C4.destroy()
        D1.destroy()
        D2.destroy()
        
    
    def reset_batch():
        label_file["text"] = "No File Selected"
        file_path = label_file["text"]
        label_cement_batch5["text"] = "Output Section"
        DATA.drop(DATA.index , inplace=True)
    
    def open_file_dialog():
        """This Function will open the file explorer and assign the chosen file path to label_file"""
        filename = filedialog.askopenfilename(initialdir="/",title="Select A File",filetype=(("csv files", ".csv"),("All Files", ".*")))
        label_file["text"] = filename
        return None
    
    def load_file_dialog():
        
        global DATA
        file_path = label_file["text"]
        # print(file_path)
        try:
#             with open(file_path, 'r',newline='') as f:
#                 data = csv.reader(f)
            DATA = pd.read_csv(file_path,header=None)
            
            
        except ValueError:
            messagebox.showerror("Information", "The file you have chosen is invalid")
            return None
        except FileNotFoundError:
            messagebox.showerror("Information", f"No such file as {file_path}")
            return None    
        
       
    def compression_batch():
        
        global batch_compressive
        
        batch_compressive = calci(DATA,"b_compressive")
    
        label_cement_batch5.config(text = " Compressive Strength (MPa) " + str(batch_compressive) + " ")
        
    def tensile_batch():
        
        global batch_tensile
        
        batch_tensile = calci(DATA,"b_tensile")
        label_cement_batch5.config(text = " Split Tensile (MPa) " + str(batch_tensile) + " ")
    
    def export_result():
        
        variable_name = "batch_compressive"

        value = globals().get(variable_name)
        
        if value is not None:
            batch_compressive2 = pd.DataFrame(batch_compressive)
            DATA2 = pd.concat([DATA, batch_compressive2], axis = 1)
            
        else:
            batch_tensile2 = pd.DataFrame(batch_tensile)
            DATA2 = pd.concat([DATA, batch_tensile2],axis = 1)
            
        try:
            files = (("Excel files", "*.xlsx"),('CSV Files','*.csv'))
            save_path = filedialog.asksaveasfilename(defaultextension=files,filetypes=files )
    
            if save_path:
                if re.search("\.xlsx$", save_path):
                   # Export DataFrame to Excel
                    DATA2.to_excel(save_path, index=False)
                
                elif re.search("\.csv$", save_path):
                    # Export DataFrame to CSV
                    DATA2.to_csv(save_path,index=False)
            



        except FileNotFoundError:
            tk.messagebox.showerror("Information", "Please upload the file first")
            return None          

            
        
        
    # The file/file path text
    label_file = Label(root, text="No File Selected",width = 100)
    label_file.place(x=10,y=260)

    
    label_cement_batch = Label(root,text="Please upload your file in csv format",width = 40,font=("bold",15))
    label_cement_batch.place(x=5,y=130)
    
    label_cement_batch2 = Label(root,text="[Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate,Age] for compressive Mpa ( Age in days, remaining in Kg/m^3) ",width = 130,font=("bold",10))
    label_cement_batch2.place(x=5,y=160)
    
    label_cement_batch3 = Label(root,text="[Compressive strength of cement fce (MPa),Tensile strength of cement fct (MPa),Curing age (day),Dmax of crushed stone (mm),Stone powder content in sand (%), ",width = 133,font=("bold",10))
    label_cement_batch3.place(x=5,y=200)
    
    label_cement_batch4 = Label(root,text=" Fineness modulus of sand-W/B,Water to cement ratio-mw/mc,Water (kg/m3),Sand ratio (%),Slump (mm),Compressive strength, fcu,t (MPa)] for tensile Mpa ",width = 128,font=("bold",10))
    label_cement_batch4.place(x=5,y=220)
    
    label_cement_batch5 = Label(root,text=" Output Section ",width = 100,font=("bold",15))
    label_cement_batch5.place(x=5,y=500)
    
    B5=Button(root,text = "Compressive",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:compression_batch())
    B5.place(x=200,y=400)
    B7=Button(root,text = "Tensile",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:tensile_batch())
    B7.place(x=400,y=400)
    B6=Button(root,text = "Reset Data",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:reset_batch())
    B6.place(x=550,y=400)
    C3=Button(root,text = "Clear Screen",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:clear_batch())
    C3.place(x=730,y=400)
    C4=Button(root,text = "Export",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:export_result())
    C4.place(x=930,y=400)
    D1=Button(root,text = "Open File",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:open_file_dialog())
    D1.place(x=550,y=300)
    D2=Button(root,text = "Load File",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:load_file_dialog())
    D2.place(x=750,y=300)
    





root.mainloop()

