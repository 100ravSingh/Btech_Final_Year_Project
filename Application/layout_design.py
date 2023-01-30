#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
import main as app





root = Tk()
root.geometry("1000x800")
root.title('HPC Strength Calculator')

label_0 = Label(root,text="VSSUT HPC Strength Calculator",width = 40,font=("bold",20))
label_0.place(x=150,y=30)


Button(root,text = "Compressive_Page",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_mycomp()).place(x=350,y=80)
Button(root,text = "Tensile_Page",highlightbackground="black",justify = CENTER,font = ('courier', 10, 'bold'),bg='cyan',pady=6,command=lambda:set_myten()).place(x=550,y=80)

    
    


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
    
        compressive = app.calci(T1,"compressive")
    
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
    
        tensile = app.calci(T1,"tensile")
    
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
    
    














# In[2]:


root.mainloop()

