import dit
import admUI
import math
import pandas as pd
import numpy as np
###########Ingest#############
output_path = "C:/Users/ginge/Desktop/Output/"
data_url = "diabetes.csv"
column_names = ["Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
                "Outcome"]
df_full = pd.read_csv(data_url, names=column_names,skiprows=1)
unique_A = {}
unique_B = {}
name_a = "Age"
name_b = "BMI"
#trials = [1,10,50,100,300,600]
trials = [df_full.shape[0]]
sims = 1
##########Discretize###########
for i in trials:
 a = []
 b = []
 for _ in range(sims):
  df = df_full.sample(n=i)
  bmi_groups = ["<18","18-24", "25-30",">30"]
  df["bmi-groups"] = df.BMI.apply(lambda x: bmi_groups[np.digitize(x,[18,25,30])])
  age_groups = ["<24","24-35","36-50",">50"]
  df["age-groups"] = df.Age.apply(lambda x: age_groups[np.digitize(x,[24,36,51])])
##########Isolate###########
  selected_columns = ["Outcome", "age-groups", "bmi-groups"]
  rvs_names = ["O","A","B"]
  rvs_to_name = dict(zip(rvs_names,selected_columns))
  data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))
  dist_census = dit.Distribution(data_array, [1. / df.shape[0]] * df.shape[0])
  dist_census.set_rv_names("".join(rvs_names))
##########Optimize###########
  q_OBA = admUI.computeQUI(distSXY = dist_census.marginal('OBA'))
  q_OBA.set_rv_names("OBA")
##########Decompose###########

#H(O|B,A), Conditional entopy
  h_OBA =  dit.shannon.conditional_entropy(q_OBA, 'O', 'BA')
#UI(O;A\B) = I_Q(O|B) - H(O|B,A), unique information given by A
  ui_A = dit.shannon.conditional_entropy(q_OBA, 'O', 'B') - h_OBA
  print(ui_A)
#UI(O;B\A) = I_Q(O|A) - H(O|B,A), unique information given by B
  ui_B = dit.shannon.conditional_entropy(q_OBA, 'O', 'A') - h_OBA
  print(ui_B)
#SI
  si= dit.shannon.mutual_information(q_OBA, 'O','B') - ui_B
  print(si)
#ci
  ci=si - dit.multivariate.coinformation(q_OBA, 'OBA')
  print(ci)
  a.append(ui_A)
  b.append(ui_B)

 unique_A[str(i)] = a
 unique_B[str(i)] = b
#print(f"Unique Information from A: {ui_A}")
#print(f"Unique Information from B: {ui_B}")
#print(f"Shared Information: {si}")
#print(f"Synegistic Informaton: {ci}")
#a_output = pd.DataFrame(unique_A)
#b_output = pd.DataFrame(unique_B)
#print(unique_A)
#a_output.to_csv(output_path + name_a + ".csv",index= False)
#b_output.to_csv(output_path + name_b + ".csv",index=False)
