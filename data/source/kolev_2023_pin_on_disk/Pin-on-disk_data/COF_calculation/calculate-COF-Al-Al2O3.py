# Import pandas and matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.font_manager import FontProperties
import numpy as np

# Set the font family to 'sans-serif' and the font sans-serif to 'Arial'
font = FontProperties()
font.set_style('normal')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

# Read each excel file containing the data for one dataset of one material using pd.read_excel()
# Material AC
file_AC1 = pd.read_excel("AC_3_2.xlsx") # read file for dataset A1
file_AC2 = pd.read_excel("AC_3_3.xlsx") # read file for dataset A2
file_AC3 = pd.read_excel("AC_3_4.xlsx") # read file for dataset A3

# Material C
file_C1 = pd.read_excel("C_5_2.xlsx") # read file for dataset B1
file_C2 = pd.read_excel("C_7_1.xlsx") # read file for dataset B2
file_C3 = pd.read_excel("C_5_1.xlsx") # read file for dataset B3

# Material AE
file_AE1 = pd.read_excel("AE_3_2.xlsx") # read file for dataset C1
file_AE2 = pd.read_excel("AE_4_1.xlsx") # read file for dataset C2
file_AE3 = pd.read_excel("AE_6_6.xlsx") # read file for dataset C3

# Material E
file_E1 = pd.read_excel("E_3_1.xlsx") # read file for dataset D1
file_E2 = pd.read_excel("E_6_3.xlsx") # read file for dataset D2
file_E3 = pd.read_excel("E_6.xlsx") # read file for dataset D3

# Assign each dataset to a variable using file["column name"]
# Material AC
time_AC1 = file_AC1["TIME"] # time in seconds for dataset A1
friction_AC1 = file_AC1["FF"] # friction force in N for dataset A1
time_AC2 = file_AC2["TIME"] # time in seconds for dataset A2
friction_AC2 = file_AC2["FF"] # friction force in N for dataset A2
time_AC3 = file_AC3["TIME"] # time in seconds for dataset A3
friction_AC3 = file_AC3["FF"] # friction force in N for dataset A3

# Material C
time_C1 = file_C1["TIME"] # time in seconds for dataset B1
friction_C1 = file_C1["FF"] # friction force in N for dataset B1
time_C2 = file_C2["TIME"] # time in seconds for dataset B2
friction_C2 = file_C2["FF"] # friction force in N for dataset B2
time_C3 = file_C3["TIME"] # time in seconds for dataset B3
friction_C3 = file_C3["FF"] # friction force in N for dataset B3

# Material AE
time_AE1 = file_AE1["TIME"] # time in seconds for dataset C1
friction_AE1 = file_AE1["FF"] # friction force in N for dataset C1
time_AE2 = file_AE2["TIME"] # time in seconds for dataset C2
friction_AE2 = file_AE2["FF"] # friction force in N for dataset C2
time_AE3 = file_AE3["TIME"] # time in seconds for dataset C3
friction_AE3 = file_AE3["FF"] # friction force in N for dataset C3

# Material E
time_E1 = file_E1["TIME"] # time in seconds for dataset D1
friction_E1 = file_E1["FF"] # friction force in N for dataset D1
time_E2 = file_E2["TIME"] # time in seconds for dataset D2
friction_E2 = file_E2["FF"] # friction force in N for dataset D2
time_E3 = file_E3["TIME"] # time in seconds for dataset D3
friction_E3 = file_E3["FF"] # friction force in N for dataset D3

# Calculate the coefficient of friction for each dataset using the formula: coefficient of friction = friction force / normal force

# Assume normal force is 50 N

# Material AC 
cof_AC1 = friction_AC1 / 50 # coefficient of friction for dataset A1 
cof_AC2 = friction_AC2 / 50 # coefficient of friction for dataset A2 
cof_AC3 = friction_AC3 / 50 # coefficient of friction for dataset A3 

# Material C 
cof_C1 = friction_C1 / 50 # coefficient of friction for dataset B1 
cof_C2 = friction_C2 / 50 # coefficient of friction for dataset B2 
cof_C3 = friction_C3 / 50 # coefficient of friction for dataset B3 

# Material AE 
cof_AE1 = friction_AE1 / 50 # coefficient of friction for dataset C1 
cof_AE2 = friction_AE2 / 50 # coefficient of friction for dataset C2 
cof_AE3 = friction_AE3 / 50 # coefficient of friction for dataset C3 

# Material E 
cof_E1 = friction_E1 / 50 # coefficient of friction for dataset D1 
cof_E2 = friction_E2 / 50 # coefficient of friction for dataset D2 
cof_E3 = friction_E3 / 50 # coefficient of friction for dataset D3 

# Create an empty list to store the average coefficient of friction for each material

avg_cof_AC_list= [] 
avg_cof_C_list= [] 
avg_cof_AE_list= [] 
avg_cof_E_list= [] 

# Create an empty list to store the average time for each material

avg_time_AC_list= []  
avg_time_C_list= []   
avg_time_AE_list= []  
avg_time_E_list= []   

# Use a loop to iterate over the 4 materials and append the mean value to the corresponding list 
# the variable time_A is not defined or has a different name. 
# You should use the same variable name that you used to assign the time values for dataset A1. 
# For example, if you used time_A1 = file_A1[“A1_time”], then you should use len(time_A1) instead of len(time_A). Alternatively, 
# you can use any other variable that has the same length as time_A, such as friction_A1 or cof_A1.

for i in range(0, len(time_AC1)):
    avg_cof_AC_list.append((cof_AC1[i] + cof_AC2[i] + cof_AC3[i]) / 3) 
    avg_cof_C_list.append((cof_C1[i] + cof_C2[i] + cof_C3[i]) / 3) 
    avg_cof_AE_list.append((cof_AE1[i] + cof_AE2[i] + cof_AE3[i]) / 3) 
    avg_cof_E_list.append((cof_E1[i] + cof_E2[i] + cof_E3[i]) / 3) 

# time

    avg_time_AC_list.append((time_AC1[i] + time_AC2[i] + time_AC3[i]) / 3) 
    avg_time_C_list.append((time_C1[i] + time_C2[i] + time_C3[i]) / 3)   
    avg_time_AE_list.append((time_AE1[i] + time_AE2[i] + time_AE3[i]) / 3) 
    avg_time_E_list.append((time_E1[i] + time_E2[i] + time_E3[i]) / 3)   


# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_AC_df= pd.DataFrame(avg_cof_AC_list, columns=["Average Coefficient Of Friction AC"]) 
avg_cof_AC_df.to_excel("Average Coefficient Of Friction For Material AC.xlsx", index=False) 

avg_cof_C_df= pd.DataFrame(avg_cof_C_list, columns=["Average Coefficient Of Friction C"]) 
avg_cof_C_df.to_excel("Average Coefficient Of Friction For Material C.xlsx", index=False) 

avg_cof_AE_df= pd.DataFrame(avg_cof_AE_list, columns=["Average Coefficient Of Friction AE"]) 
avg_cof_AE_df.to_excel("Average Coefficient Of Friction For Material AE.xlsx", index=False) 

avg_cof_E_df= pd.DataFrame(avg_cof_E_list, columns=["Average Coefficient Of Friction E"]) 
avg_cof_E_df.to_excel("Average Coefficient Of Friction For Material E.xlsx", index=False) 

# time

avg_time_AC_df= pd.DataFrame(avg_time_AC_list, columns=["Average Time AC"]) 
avg_time_AC_df.to_excel("Average Time For Material AC.xlsx", index=False)   

avg_time_C_df= pd.DataFrame(avg_time_C_list, columns=["Average Time C"])    
avg_time_C_df.to_excel("Average Time For Material C.xlsx", index=False)     

avg_time_AE_df= pd.DataFrame(avg_time_AE_list, columns=["Average Time AE"]) 
avg_time_AE_df.to_excel("Average Time For Material AE.xlsx", index=False)   

avg_time_E_df= pd.DataFrame(avg_time_E_list, columns=["Average Time E"]) 
avg_time_E_df.to_excel("Average Time For Material E.xlsx", index=False)  


# Apply a style sheet
plt.style.use('seaborn-darkgrid')


# Create shadow effect objects with different transparency, blur, angle and distance and smaller linewidth
pe1 = [patheffects.SimpleLineShadow(offset=(1, -1), alpha=0.4, linewidth=3), patheffects.Normal()]


# Create two plots using plt.figure() and plt.plot() with different options

fig1 = plt.figure(9) 

plt.plot(time_AC1, avg_cof_AC_list, 'red', label="Material AC", linewidth=3, alpha=0.9, zorder=4, marker=None, path_effects=pe1)  
plt.plot(time_C1, avg_cof_C_list, 'blue', label="Material C", linewidth=3, alpha=0.9, zorder=4, marker=None, path_effects=pe1)  


# Add a title and axis labels
# plt.title("Average Coefficient Of Friction vs Time For Materials AC and C")
plt.xlabel("Time, s", fontproperties=font,  fontsize='16', fontweight='bold')
plt.ylabel("Coefficient Of Friction, -", fontproperties=font,  fontsize='16', fontweight='bold')

# Add a legend with shadow and different font size 
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# Set the x axis limit to 450
plt.xlim(0, 450)

# Set the y axis limit to 0.6
plt.ylim(0, 0.6)

# Change the axes numbering size and font
plt.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')


fig2 = plt.figure(10) 

plt.plot(time_AE1, avg_cof_AE_list, 'green', label="Material AE", linewidth=3, alpha=0.9, zorder=4, marker=None, path_effects=pe1) # plot average cof as a function of time for material AE 
plt.plot(time_E1, avg_cof_E_list, 'orange', label="Material E", linewidth=3, alpha=0.9, zorder=4, marker=None, path_effects=pe1) # plot average cof as a function of time for material E 


# Add a title and axis labels
# plt.title("Average Coefficient Of Friction vs Time For Materials AE and E")
plt.xlabel("Time, s", fontproperties=font,  fontsize='16', fontweight='bold')
plt.ylabel("Coefficient Of Friction, -", fontproperties=font,  fontsize='16', fontweight='bold')

# Add a legend with shadow and different font size 
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# Set the x axis limit to 450
plt.xlim(0, 450)

# Set the y axis limit to 0.6
plt.ylim(0, 0.6)

# Change the axes numbering size and font
plt.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')


# Show both plots using plt.show()

plt.show()



# Save both plots with dpi=500 in 'png'

fig1.savefig('figure9.png', dpi=500)

fig2.savefig('figure10.png', dpi=500)






