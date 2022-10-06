import dit
import admUI
import pandas as pd
import numpy as np
import KSG_estimator as KSG
from MLE_GaussB import MLMI
import subprocess
import os


def simple_outcome(Age, Bmi, num=False):
    out = []
    for i in range(len(Bmi)):
        if Bmi[i] > 24:
            out.append(1)
        else:
            out.append(0)
    return out

def build(n=10000):
    db = pd.DataFrame()
    db["age"] = np.random.normal(35, 5, n)
    db["bmi"] = np.random.normal(24, 4, n)
    conditions = [
        (db['age'] >= 35),
        (db['age'] < 35)
    ]
    db["age-groups"] = np.select(conditions, [">=35", "<35"])
    conditions = [
        (db['bmi'] >= 24),
        (db['bmi'] < 24)
    ]
    db["bmi-groups"] = np.select(conditions, [">=24", "<24"])
    db["Outcome"] = simple_outcome(db["age"], db["bmi"])
    db.to_csv("db.csv")
    return db



def fulldat_UI(df):
    selected_columns = ["Outcome", "age-groups", "bmi-groups"]
    rvs_names = ["O", "A", "B"]
    rvs_to_name = dict(zip(rvs_names, selected_columns))
    data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))
    dist_census = dit.Distribution(data_array, [1. / df.shape[0]] * df.shape[0])
    dist_census.set_rv_names("".join(rvs_names))
    ##########Optimize###########
    q_OBA = admUI.computeQUI(distSXY=dist_census.marginal('OBA'))
    q_OBA.set_rv_names("OBA")
    ##########Decompose###########

    # H(O|B,A), Conditional entopy
    h_OBA = dit.shannon.conditional_entropy(q_OBA, 'O', 'BA')
    # UI(O;A\B) = I_Q(O|B) - H(O|B,A), unique information given by A
    ui_B = dit.shannon.conditional_entropy(q_OBA, 'O', 'B') - h_OBA
    # UI(O;B\A) = I_Q(O|A) - H(O|B,A), unique information given by B
    ui_A = dit.shannon.conditional_entropy(q_OBA, 'O', 'A') - h_OBA
    # SI
    si = dit.shannon.mutual_information(q_OBA, 'O', 'B') - ui_B
    # ci
    ci = si - dit.multivariate.coinformation(q_OBA, 'OBA')
    return [ui_B, ui_A]

def fulldat_mi(df):
    mi_A = KSG.revised_mi(np.array([df["age"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
    mi_B = KSG.revised_mi(np.array([df["bmi"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
    return [mi_B, mi_A]


def ui_replicate(dat, trueUI, reps=50, trials=[10, 100, 500, 1000, 5000], epsilon=1e-3):
    out_A = [0.] * len(trials)
    out_B = [0.] * len(trials)
    for i in range(len(trials)):
        a = []
        b = []
        for _ in range(reps):
            df = dat.sample(n=trials[i])
            ##########Isolate###########
            selected_columns = ["Outcome", "age-groups", "bmi-groups"]
            rvs_names = ["O", "A", "B"]
            rvs_to_name = dict(zip(rvs_names, selected_columns))
            data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))
            dist_census = dit.Distribution(data_array, [1. / df.shape[0]] * df.shape[0])
            dist_census.set_rv_names("".join(rvs_names))
            ##########Optimize###########
            q_OBA = admUI.computeQUI(distSXY=dist_census.marginal('OBA'))
            q_OBA.set_rv_names("OBA")
            ##########Decompose###########
            # H(O|B,A), Conditional entopy
            h_OBA = dit.shannon.conditional_entropy(q_OBA, 'O', 'BA')
            # UI(O;A\B) = I_Q(O|B) - H(O|B,A), unique information given by A
            ui_B = dit.shannon.conditional_entropy(q_OBA, 'O', 'B') - h_OBA
            # UI(O;B\A) = I_Q(O|A) - H(O|B,A), unique information given by B
            ui_A = dit.shannon.conditional_entropy(q_OBA, 'O', 'A') - h_OBA
            # SI
            si = dit.shannon.mutual_information(q_OBA, 'O', 'B') - ui_B
            # ci
            ci = si - dit.multivariate.coinformation(q_OBA, 'OBA')
            if abs(ui_A - trueUI[1]) < epsilon:
                a.append(1)
            else:
                a.append(0)
            if abs(ui_B - trueUI[0]) < epsilon:
                b.append(1)
            else:
                b.append(0)
        out_A[i] = np.mean(a)
        out_B[i] = np.mean(b)
    return (out_A, out_B)

def mi_replicate(dat, trueMI, reps=50, trials=[10, 100, 500, 1000, 5000], epsilon=1e-3):
    out_A = [0.] * len(trials)
    out_B = [0.] * len(trials)
    for i in range(len(trials)):
        a = []
        b = []
        for _ in range(reps):
            df = dat.sample(n=trials[i])
            mi_A = KSG.revised_mi(np.array([df["age"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
            mi_B = KSG.revised_mi(np.array([df["bmi"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
            if abs(mi_A - trueMI[1]) < epsilon:
                a.append(1)
            else:
                a.append(0)
            if abs(mi_B - trueMI[0]) < epsilon:
                b.append(1)
            else:
                b.append(0)
        out_A[i] = np.mean(a)
        out_B[i] = np.mean(b)
    return (out_A, out_B)

def mivsui(dat, trueMI, trueUI, reps=50, trials=[10, 100, 500, 1000, 5000], epsilon=1e-3):
    out_AUI = [0.] * len(trials)
    out_BUI = [0.] * len(trials)
    out_AMI = [0.] * len(trials)
    out_BMI = [0.] * len(trials)
    for i in range(len(trials)):
        a = []
        b = []
        a2 = []
        b2 = []
        for _ in range(reps):
            df = dat.sample(n=trials[i])
            ##########Isolate###########
            selected_columns = ["Outcome", "age-groups", "bmi-groups"]
            rvs_names = ["O", "A", "B"]
            rvs_to_name = dict(zip(rvs_names, selected_columns))
            data_array = list(map(lambda r: tuple(r[k] for k in selected_columns), df.to_dict("record")))
            dist_census = dit.Distribution(data_array, [1. / df.shape[0]] * df.shape[0])
            dist_census.set_rv_names("".join(rvs_names))
            ##########Optimize###########
            q_OBA = admUI.computeQUI(distSXY=dist_census.marginal('OBA'))
            q_OBA.set_rv_names("OBA")
            ##########Decompose###########
            # H(O|B,A), Conditional entopy
            h_OBA = dit.shannon.conditional_entropy(q_OBA, 'O', 'BA')
            # UI(O;A\B) = I_Q(O|B) - H(O|B,A), unique information given by A
            ui_B = dit.shannon.conditional_entropy(q_OBA, 'O', 'B') - h_OBA
            # UI(O;B\A) = I_Q(O|A) - H(O|B,A), unique information given by B
            ui_A = dit.shannon.conditional_entropy(q_OBA, 'O', 'A') - h_OBA
            # SI
            si = dit.shannon.mutual_information(q_OBA, 'O', 'B') - ui_B
            # ci
            ci = si - dit.multivariate.coinformation(q_OBA, 'OBA')
            if abs(ui_A - trueUI[1]) < epsilon:
                a.append(1)
            else:
                a.append(0)
            if abs(ui_B - trueUI[0]) < epsilon:
                b.append(1)
            else:
                b.append(0)
            mi_A = KSG.revised_mi(np.array([df["age"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
            mi_B = KSG.revised_mi(np.array([df["bmi"]]).T.tolist(), np.array([df["Outcome"]]).T.tolist())
            if abs(mi_A - trueMI[1]) < epsilon:
                a2.append(1)
            else:
                a2.append(0)
            if abs(mi_B - trueMI[0]) < epsilon:
                b2.append(1)
            else:
                b2.append(0)
        out_AUI[i] = np.mean(a)
        out_BUI[i] = np.mean(b)
        out_AMI[i] = np.mean(a2)
        out_BMI[i] = np.mean(b2)
    out_B = pd.DataFrame()
    out_B["MI"] = out_BMI
    out_B["UI"] = out_BUI
    out_A = pd.DataFrame()
    out_A["MI"] = out_AMI
    out_A["UI"] = out_AUI
    out_B.to_csv("BMI.csv")
    out_B.to_csv("Age.csv")
    return ([out_AUI, out_BUI], [out_AMI, out_BMI])


if __name__ == '__main__':
    direc = "C:/Users/Scott/OneDrive/Documents/mi_ui/UIsim_Pima"
    db = build()
    print(mivsui(db, fulldat_mi(db), fulldat_UI(db)))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
