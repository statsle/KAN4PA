import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from scipy.stats import norm, t
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from dataset_dry import SignificantFiguresCounter
from kan import *
import logging
import argparse

def calculate_auc(actual, predicted):
    
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(actual, predicted)
    
    sorted_indices = np.argsort(predicted)[::-1]
    actual_sorted = actual[sorted_indices]
    predicted_sorted = predicted[sorted_indices]
    
    return auc_score


def process_DrySpotBloodDateset(train=True, transform = None, target_transform=None, data=None, the_select_list=None):
    train = train
    data = data
    transform = transform
    target_transform = target_transform

    feature = []
    labels = []
    selected_indices = the_select_list
    
    for i in range(len(data)):
        selected_values = [data[i][idx] for idx in selected_indices]
            
        feature.append(selected_values)
        labels.append(data[i][1])  
        
    return feature, labels


def stat_data(data=None):
    tot_1, tot_0, tot_2, animia_1, animia_2, animia_0 = 0, 0, 0, 0, 0, 0
    for i in range(len(data)):
        data_ = data[i]
        if data_[29] == 0 :
            tot_0 += 1
            if data_[1] == 1:
                animia_0 += 1
        elif data_[29] == 1 :
            tot_1 += 1
            if data_[1] == 1:
                animia_1 += 1            
        elif data_[29] == 2 :
            tot_2 += 1       
            if data_[1] == 1:
                animia_2 += 1             
    print(f"iron_type==0={tot_0/len(data)}\n iron_type==1={tot_1/len(data)}\n iron_type==2={tot_2/len(data)}\n iron_type==0 and animia={animia_0/tot_0}\n iron_type==1 and animia={animia_1/tot_1}\n iron_type==2 and animia={animia_2/tot_2}\n")    


def write_xlsx_age(data=None, age_list=None, type_=None):
    catagory = []
    print(len(age_list))
    print(age_list)
    
    for i in range(0, 257):
        num = 0
        if age_list[i] == 0:
            num = -1
        elif age_list[i] < 20:
            num = 0
        elif age_list[i] < 25:
            num = 1     
        elif age_list[i] < 30:
            num = 2
        elif age_list[i] < 35:
            num = 3  
        elif age_list[i] < 40:
            num = 4
        else :
            num = 5
        catagory.append(num)            
    
    data = {'age': catagory}
    df = pd.DataFrame(data)
    
    with pd.ExcelWriter('example.xlsx', engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='temp', startrow=1, startcol=3, index=False)


def svm(train_data_normalized=None, valid_data_normalized=None, the_select_list=None):
    t_feature = []
    t_labels = []
    v_feature = []
    v_labels = []        
    selected_indices = the_select_list
    
    for i in range(len(train_data_normalized)):
        selected_values = [train_data_normalized[i][idx] for idx in selected_indices]
        t_feature.append(selected_values)
        t_labels.append(train_data_normalized[i][1])  
        
    for i in range(len(valid_data_normalized)):
        selected_values = [valid_data_normalized[i][idx] for idx in selected_indices]
        v_feature.append(selected_values)
        v_labels.append(valid_data_normalized[i][1])  
    
    svm = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    svm.fit(t_feature, t_labels)

    y_pred_t = svm.predict_proba(t_feature)[:, -1]
    y_pred_v = svm.predict_proba(v_feature)[:, -1]
    
    return y_pred_t, y_pred_v


def naive_bys(train_data_normalized=None, valid_data_normalized=None, the_select_list=None):
    t_feature = []
    t_labels = []
    v_feature = []
    v_labels = []        
    selected_indices = the_select_list
    
    for i in range(len(train_data_normalized)):
        selected_values = [train_data_normalized[i][idx] for idx in selected_indices]
        t_feature.append(selected_values)
        t_labels.append(train_data_normalized[i][1])  
        
    for i in range(len(valid_data_normalized)):
        selected_values = [valid_data_normalized[i][idx] for idx in selected_indices]
        v_feature.append(selected_values)
        v_labels.append(valid_data_normalized[i][1])  
    
    nb_classifier = GaussianNB()
    nb_classifier.fit(t_feature, t_labels)
    
    t_pred_proba = nb_classifier.predict_proba(t_feature)[:, 1]
    v_pred_proba = nb_classifier.predict_proba(v_feature)[:, 1]
    
    return t_pred_proba, v_pred_proba


def logistic_regression(train_data_normalized=None, valid_data_normalized=None, the_select_list=None):
    t_feature = []
    t_labels = []
    v_feature = []
    v_labels = []        
    selected_indices = the_select_list
    
    for i in range(len(train_data_normalized)):
        selected_values = [train_data_normalized[i][idx] for idx in selected_indices]
        if selected_values[1] != 0:
            selected_values[1] = 1
        if selected_values[-1] != 0:
            selected_values[1] = 1
        t_feature.append(selected_values)
        t_labels.append(train_data_normalized[i][1])  
        
    for i in range(len(valid_data_normalized)):
        selected_values = [valid_data_normalized[i][idx] for idx in selected_indices]
        if selected_values[1] != 0:
            selected_values[1] = 1
        if selected_values[-1] != 0:
            selected_values[1] = 1
        v_feature.append(selected_values)
        v_labels.append(valid_data_normalized[i][1])  
    
    model = LogisticRegression(random_state=42)
    model.fit(t_feature, t_labels)

    t_pred_proba = model.predict_proba(t_feature)[:, 1]
    v_pred_proba = model.predict_proba(v_feature)[:, 1]
    
    return t_pred_proba, v_pred_proba


def delong_roc_ci(y_true, y_pred):
    n = len(y_true)
    AUC = roc_auc_score(y_true, y_pred)
    

    n_bootstraps = 1000
    auc_scores = []

    if AUC > 0.8:
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_bootstrap = y_true[indices]
            y_scores_bootstrap = y_pred[indices]
            auc = roc_auc_score(y_true_bootstrap, y_scores_bootstrap)
            auc_scores.append(auc)

        confidence_interval = np.percentile(auc_scores, [2.5, 97.5])

        #print("95%置信区间:", confidence_interval)
        
        return AUC, confidence_interval
    else :
        return AUC, 0


def calculate_p_value(auc, n):
    z_stat = (auc - 0.5) / np.sqrt((auc * (1 - auc)) / n)
    p_value = norm.sf(abs(z_stat)) 
    return p_value

def calculate_p_value_nri_idi(auc, n):
    z_stat = (auc - 0.0) / np.sqrt((auc * (1 - auc)) / n)
    p_value = norm.sf(abs(z_stat)) 
    return p_value

def auc_mean_ci(auc_scores, confidence=0.95):
    
    auc_mean = np.mean(auc_scores)
    
    auc_std = np.std(auc_scores, ddof=1) 
    
    dof = len(auc_scores) - 1
    
    alpha = 1 - confidence
    t_crit = np.abs(t.ppf(alpha/2, dof))
    
    auc_ci = (auc_mean - t_crit * auc_std / np.sqrt(len(auc_scores)),
              auc_mean + t_crit * auc_std / np.sqrt(len(auc_scores)))
    
    return auc_mean, auc_ci

def main():
    
    parser = argparse.ArgumentParser(description="To evaluate the model")
    parser.add_argument("--data_site", type=str, default="toy_dataset.xlsx", help="Path to the dataset")
    parser.add_argument("--random_number", type=int, default=42, help="Random number seed")
    parser.add_argument("--N", type=int, default=10, help="Cross-validation rounds")
    parser.add_argument("--width", type=int, default=8, help="Width for KAN")
    parser.add_argument("--grid", type=int, default=3, help="grid for KAN")
    parser.add_argument("--k", type=int, default=3, help="k for KAN")
    
    args = parser.parse_args()
    
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('log.log')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    site = args.data_site
    counter = SignificantFiguresCounter(site)
    data, age = counter.count_significant_figures()
    
    torch.manual_seed(args.random_number)
    selected_indices = [2, 3, 4, 5, 6, 11, 12, 14]
    width_forkan = [len(selected_indices), args.width, 1]
    
    logger.info(f'selected_indices = {selected_indices}')
    logger.info(f'width_for_kan = {width_forkan}')
    
    for turn_ in tqdm(range(args.N)):
        kf = KFold(n_splits=5, shuffle=True, random_state=turn_)
        n = len(data)
        AUC_list_in_cross_val = []
        k_f = 0

        for train_index, valid_index in kf.split(range(n)):
            k_f += 1
            
            best_test_label = None
            best_test_pred = None
            
            train_data = [data[i] for i in train_index]
            valid_data = [data[i] for i in valid_index]
        
            # print("train_data_size:", len(train_data))
            # print("val_data_size:", len(valid_data))
            n_val = len(valid_data)
            
            scaler = StandardScaler()
            scaler.fit([sample[2:6] for sample in train_data])
            
            train_data_normalized = [[sample[0], sample[1], sample[6]*5, sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], \
                                    ] + list(scaler.transform([sample[2:6]])[0]) for sample in train_data]
            valid_data_normalized = [[sample[0], sample[1], sample[6]*5, sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], \
                                    ] + list(scaler.transform([sample[2:6]])[0]) for sample in valid_data]
            
            
            y_pred_t, y_pred_v = svm(train_data_normalized=train_data_normalized, valid_data_normalized=valid_data_normalized, the_select_list=selected_indices)
            y_pred_t_s, y_pred_v_s = naive_bys(train_data_normalized=train_data_normalized, valid_data_normalized=valid_data_normalized, the_select_list=selected_indices)
            y_pred_t_l, y_pred_v_l = logistic_regression(train_data_normalized=train_data_normalized, valid_data_normalized=valid_data_normalized, the_select_list=selected_indices)
            
            train_input, train_label = process_DrySpotBloodDateset(data=train_data_normalized, the_select_list=selected_indices)
            test_input, test_label = process_DrySpotBloodDateset(data=valid_data_normalized, the_select_list=selected_indices)

            dataset = {}
            dataset['train_input'] = torch.tensor(train_input).float()
            dataset['test_input'] = torch.tensor(test_input).float()
            dataset['train_label'] = torch.tensor(train_label).unsqueeze(1).float()
            dataset['test_label'] = torch.tensor(test_label).unsqueeze(1).float()
            
            model = KAN(width=width_forkan, grid=args.grid, k=args.k)
            
            def train_acc():
                return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).float())

            def test_acc():
                return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).float())

            auc_score = 0
            
            for train_iter in range(3):
                results = model.train(dataset, opt="LBFGS", steps=1, metrics=(train_acc, test_acc))
                
                test_label = dataset['test_label'][:,0]
                test_pred = model(dataset['test_input'])[:,0]
                
                now_auc = roc_auc_score(test_label.detach().numpy(), test_pred.detach().numpy())
                if now_auc > auc_score:
                    auc_score = now_auc
                    torch.save(model.state_dict(), f'model_mp/kan_temp.pth')
                    best_test_label = test_label.detach().numpy()
                    best_test_pred = test_pred.detach().numpy()
                    
            AUC_list_in_cross_val.append(auc_score)
            if len(AUC_list_in_cross_val) == 5:
                logger.info(f'Turn = {turn_} , AUC_list_in_cross_val for  = {AUC_list_in_cross_val}')
                print(f"\n THE MEAN AUC == {sum(AUC_list_in_cross_val) / len(AUC_list_in_cross_val)} \n")
                logger.info(f'Mean auc = {sum(AUC_list_in_cross_val) / len(AUC_list_in_cross_val)}')

            model_site = 'model_mp/kan_temp.pth'
            model.load_state_dict(torch.load(model_site))
            torch.save(model.state_dict(), f'model_mp/kan_{turn_}_{k_f}.pth')
    
    
    print(f"Training completed")


if __name__ == "__main__":
    main()