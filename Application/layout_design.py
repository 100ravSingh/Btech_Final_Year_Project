#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
import main as app


# In[2]:


root = Tk()


# In[3]:


root.geometry("1000x800")


# In[4]:


root.title('HPC Strength Calculator')


# In[5]:


label_0 = Label(root,text="VSSUT HPC Strength Calculator",width = 40,font=("bold",20))
label_0.place(x=150,y=40)


# In[6]:


label_cement = Label(root,text="Cement (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_cement.place(x=5,y=130)


# In[7]:


cement_1 = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
cement_1.place(x=600,y=130)


# In[8]:


label_BF = Label(root,text="BF Slag (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_BF.place(x=5,y=170)


# In[9]:


BF_1 = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
BF_1.place(x=600,y=170)


# In[10]:


label_ash = Label(root,text="Fly ash (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_ash.place(x=5,y=210)


# In[11]:


fly_ash = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
fly_ash.place(x=600,y=210)


# In[12]:


label_water = Label(root,text="Water (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_water.place(x=5,y=250)


# In[13]:


water = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
water.place(x=600,y=250)


# In[14]:


label_plasticizer = Label(root,text="Superplasticizer (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_plasticizer.place(x=5,y=290)


# In[15]:


plasticizer = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
plasticizer.place(x=600,y=290)


# In[16]:


label_coarse = Label(root,text="Coarse Aggregrate (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_coarse.place(x=5,y=330)


# In[17]:


coarse = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
coarse.place(x=600,y=330)


# In[18]:


label_fine = Label(root,text="Fine Aggregrate (Kg per m^3 mixture)",width = 40,font=("bold",15))
label_fine.place(x=5,y=370)


# In[19]:


fine = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
fine.place(x=600,y=370)


# In[20]:


label_age = Label(root,text="Age ( Day )",width = 40,font=("bold",15))
label_age.place(x=5,y=410)


# In[21]:


day = Entry(root,highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'))
day.place(x=600,y=410)


# In[22]:


label_comp = Label(root,text="",width = 40,font=("bold",15))
label_comp.place(x=200,y=625)


# In[23]:


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
    
    


# In[24]:


#fields=[cement_1,BF_1,fly_ash,water,plasticizer,coarse,fine,day]
def reset_data():
    
    cement_1.delete(first=0,last=15)
    BF_1.delete(0,END)
    fly_ash.delete(0,END)
    water.delete(0,END)
    plasticizer.delete(0,END)
    coarse.delete(0,END)
    fine.delete(0,END)
    day.delete(0,END)
    label_comp.config(text="")
    


# In[25]:


#border=Frame(root,highlightbackground="black",highlightthickness=2,bd=0)
Button(root,text = "Compressive",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:compression_data()).place(x=300,y=500)
#Button(root,text = "Tensile",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:tensile_data()).place(x=550,y=500)
Button(root,text = "Reset",highlightbackground="black",highlightthickness=2,justify = CENTER,font = ('courier', 15, 'bold'),bg='cyan',pady=10,command=lambda:reset_data()).place(x=800,y=500)


# In[26]:


root.mainloop()


# In[ ]:




