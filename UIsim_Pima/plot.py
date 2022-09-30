import matplotlib.pyplot as plt
import pandas as pd

A = pd.read_csv("C:/Users/ginge/Desktop/Output/Age.csv")
B = pd.read_csv("C:/Users/ginge/Desktop/Output/BMI.csv")
columns_list = list(A)
x_A = [float(i) for i in columns_list]
x_B = x_A
y_B = []
y_A = []
sd_A = []
sd_B = []

for i in columns_list:
  y_A.append(A[i].mean())
  y_B.append(B[i].mean())
  sd_A.append(A[i].std())
  sd_B.append(B[i].std())
plt.errorbar(x_A,y_A,yerr=sd_A, color = "black")
plt.suptitle("Unique Information - Age \n sample size = [1,10,50,100,300,600]")
plt.show()
plt.errorbar(x_B, y_B,yerr=sd_B, color = "black")
plt.suptitle("Unique Information - BMI \n sample size = [1,10,50,100,300,600]")
plt.show()