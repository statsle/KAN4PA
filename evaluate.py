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


def process_DrySpotBloodDateset(root=None, train=True, transform = None, target_transform=None, data=None, y_pred_svm=None, y_pred_naive=None, y_pred_logre=None, model_using=None, the_select_list=None):
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
            


def Count_significant_figures(site):
    data = pd.read_excel(site, sheet_name='data_mp')
    tot = 0
    tot_1 = 0
    significant_figures = []
    age = []
    
    for row in range(1, 258):
        age.append(data.iloc[row, 10])
        
        flag = 1
        row_data = []
        for col in range(15):
            selected_value = data.iloc[row, col]
            row_data.append(selected_value)
            if pd.isna(selected_value):
                flag = 0
            # else:
            #     if selected_value == -1:
            #         flag = 0
                    
        if flag == 1:        
            if row_data[1] == 1:
                tot_1 += 1
            #print(row_data[39])
            for i in range(1,6):
                onehot_ = 1.0 if contains_digit(number=row_data[10], digit=i) == True else 0.0
                row_data.append(onehot_)
            for i in range(1,6):
                onehot_ = 1.0 if contains_digit(number=row_data[13], digit=i) == True else 0.0
                row_data.append(onehot_)
            significant_figures.append(row_data)
        else :
            #print(row_data[0])
            pass
        tot += flag
    print(f'total_significant_figures={tot}')
    print(f'label=1={tot_1}')
    
    return significant_figures , age



def contains_digit(number, digit):
    number_str = str(number)
    digit_str = str(digit)
    return digit_str in number_str


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

from sklearn.utils import check_random_state

def continuous_calibration_curve(y_true, y_prob, n_bins=10, random_state=None):
    rng = check_random_state(random_state)
    thresholds = np.linspace(0, 1, n_bins + 1)
    counts = np.zeros_like(thresholds)
    sums = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        true_positives = np.logical_and(y_true == 1, y_prob >= threshold).sum()
        false_positives = np.logical_and(y_true == 0, y_prob >= threshold).sum()
        counts[i] = true_positives + false_positives
        sums[i] = true_positives
    nonempty_bins = counts > 0
    fraction_of_positives = sums[nonempty_bins] / counts[nonempty_bins]
    mean_predicted_value = thresholds[nonempty_bins]
    return fraction_of_positives, mean_predicted_value

def calculate_nri(event, base_probs, new_probs, risk_categories=[0.2, 0.4, 0.6, 0.8]):

    n_events = np.sum(event)
    n_non_events = len(event) - n_events
    
    # Calculate reclassification proportions for events and non-events
    event_reclassification = _calculate_reclassification(event, base_probs, new_probs, risk_categories)
    non_event_reclassification = _calculate_reclassification(1 - event, 1 - base_probs, 1 - new_probs, risk_categories)
    
    # Calculate NRI
    nri = (event_reclassification['up'] - event_reclassification['down']) / n_events - \
          (non_event_reclassification['down'] - non_event_reclassification['up']) / n_non_events
    
    return nri

def calculate_idi(event, base_probs, new_probs):

    # Calculate mean predicted probabilities for events and non-events
    base_event_mean = np.mean(base_probs[event == 1])
    base_non_event_mean = np.mean(base_probs[event == 0])
    new_event_mean = np.mean(new_probs[event == 1])
    new_non_event_mean = np.mean(new_probs[event == 0])
    
    # Calculate IDI
    idi = (new_event_mean - new_non_event_mean) - (base_event_mean - base_non_event_mean)
    
    return idi

def _calculate_reclassification(event, base_probs, new_probs, risk_categories):

    # Categorize predicted probabilities into risk categories
    base_risk_categories = np.digitize(base_probs, risk_categories)
    new_risk_categories = np.digitize(new_probs, risk_categories)
    
    # Calculate reclassification proportions
    reclassification = {'up': 0, 'down': 0}
    for i in range(len(event)):
        if new_risk_categories[i] > base_risk_categories[i]:
            reclassification['up'] += event[i]
        elif new_risk_categories[i] < base_risk_categories[i]:
            reclassification['down'] += event[i]
    
    return reclassification

import numpy as np
from scipy.stats import t, norm

def get_ci_p(value_list, confidence=0.95):
    
    mean = np.mean(value_list)
    auc_std = np.std(value_list, ddof=1) 
    dof = len(value_list) - 1
    alpha = 1 - confidence
    t_crit = np.abs(t.ppf(alpha/2, dof))
    ci = (mean - t_crit * auc_std / np.sqrt(len(value_list)),
              mean + t_crit * auc_std / np.sqrt(len(value_list)))
    p_value = calculate_p_value_nri_idi(mean, n=50)
    
    return mean, ci, p_value


from sklearn.metrics import confusion_matrix

def net_benefit(y_true, y_pred, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    benefit = ((tp - fp) / (tn + fp + fn + tp))
    harm = -fp / (tn + fp + fn + tp)
    net_benefit = benefit - harm * threshold
    return net_benefit


def evaluate():
    
    parser = argparse.ArgumentParser(description="To evaluate the model")
    parser.add_argument("--dca", action="store_true", default=False, help="Draw DCA graph")
    parser.add_argument("--cur", action="store_true", default=False, help="Draw Calibration curves")
    parser.add_argument("--pruning", action="store_true", default=False, help="Prune the model")
    parser.add_argument("--symbo_regre", action="store_true", default=False, help="Perform symbolic regression")
    parser.add_argument("--data_site", type=str, default="toy_dataset.xlsx", help="Path to the dataset")
    parser.add_argument("--random_number", type=int, default=42, help="Random number seed")
    parser.add_argument("--N", type=int, default=10, help="Cross-validation rounds")
    parser.add_argument("--width", type=int, default=8, help="Width for KAN")

    args = parser.parse_args()
    
    site = args.data_site
    counter = SignificantFiguresCounter(site)
    data, age = counter.count_significant_figures()
    
    torch.manual_seed(args.random_number)
    
    mean_auc = 0.0          # Mean Area Under the Curve
    mean_accuracy = 0.0     # Mean Accuracy
    mean_sensitivity = 0.0  # Mean Sensitivity (True Positive Rate)
    mean_specificity = 0.0  # Mean Specificity (True Negative Rate)
    AUC_list_per_cross_val = []
    tot_actual_labels_val = np.array([])
    tot_predicted_probs_val = np.array([])
    tot_actual_labels_tra = np.array([])
    tot_predicted_probs_tra = np.array([])
    tot_base_probs_tra = np.array([])
    tot_base_probs_val = np.array([])
    nri_train_list = []
    nri_test_list = []
    idi_train_list = []
    idi_test_list = []
    
    # Selected Factors included in the model
    selected_indices = [2, 3, 4, 5, 6, 11, 12, 14]
    selected_indices_for_logis = [2, 3, 4, 5, 6, 7]
    selected_indices_for_mptra = [2, 3, 4, 5, 6, 11, 12, 14, 7, 8]
    svm_indices = [2, 3, 4, 11, 12, 14]
    
    width_forkan = [len(selected_indices), args.width, 1]
    
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
            
            train_data_normalized_for_logist = [[sample[0], sample[1], sample[6], sample[7], sample[8], sample[9], sample[10], sample[15], sample[11], sample[12], sample[13], sample[14], \
                                    ] + list(scaler.transform([sample[2:6]])[0]) for sample in train_data]
            valid_data_normalized_for_logist = [[sample[0], sample[1], sample[6], sample[7], sample[8], sample[9], sample[10], sample[15], sample[11], sample[12], sample[13], sample[14], \
                                    ] + list(scaler.transform([sample[2:6]])[0]) for sample in valid_data]
            
            
            y_pred_t, y_pred_v = svm(train_data_normalized=train_data_normalized, valid_data_normalized=valid_data_normalized, the_select_list=svm_indices)
            y_pred_t_s, y_pred_v_s = naive_bys(train_data_normalized=train_data_normalized, valid_data_normalized=valid_data_normalized, the_select_list=selected_indices)
            y_pred_t_l, y_pred_v_l = logistic_regression(train_data_normalized=train_data_normalized_for_logist, valid_data_normalized=valid_data_normalized_for_logist, the_select_list=selected_indices_for_logis)
            
            tot_base_probs_tra = np.concatenate((tot_base_probs_tra, y_pred_t_l))
            tot_base_probs_val = np.concatenate((tot_base_probs_val, y_pred_v_l))
            
            train_input, train_label = process_DrySpotBloodDateset(data=train_data_normalized, y_pred_svm=y_pred_t, y_pred_naive=y_pred_t_s, y_pred_logre=y_pred_t_l, model_using="svm", the_select_list=selected_indices)
            test_input, test_label = process_DrySpotBloodDateset(data=valid_data_normalized, y_pred_svm=y_pred_v, y_pred_naive=y_pred_v_s, y_pred_logre=y_pred_v_l, model_using="svm", the_select_list=selected_indices)
            
            dataset = {}
            dataset['train_input'] = torch.tensor(train_input).float()
            dataset['test_input'] = torch.tensor(test_input).float()
            dataset['train_label'] = torch.tensor(train_label).unsqueeze(1).float()
            dataset['test_label'] = torch.tensor(test_label).unsqueeze(1).float()
            
            model = KAN(width=width_forkan, grid=3, k=3)
            
            model_site = f'model_mp/kan_{turn_}_{k_f}.pth'
            model.load_state_dict(torch.load(model_site))
            
            best_test_label = dataset['test_label'][:,0].detach().numpy()
            best_test_pred = model(dataset['test_input'])[:,0].detach().numpy()
            best_train_label = dataset['train_label'][:,0].detach().numpy()
            best_train_pred = model(dataset['train_input'])[:,0].detach().numpy()

            nri_train = calculate_nri(event=best_train_label, base_probs=y_pred_t_l, new_probs=best_train_pred)
            idi_train = calculate_idi(event=best_train_label, base_probs=y_pred_t_l, new_probs=best_train_pred)
            
            nri_test = calculate_nri(event=best_test_label, base_probs=y_pred_v_l, new_probs=best_test_pred)
            idi_test = calculate_idi(event=best_test_label, base_probs=y_pred_v_l, new_probs=best_test_pred)
            
            nri_train_list.append(nri_train)
            idi_train_list.append(idi_train)
            nri_test_list.append(nri_test)
            idi_test_list.append(idi_test)
            
            if turn_ == 0 and args.pruning == True and k_f == 1:
                model.plot(site_img='model')
                model.prune()
                model.plot(mask=True, site_img='pruned_model_1')
                model = model.prune()
                model(dataset['train_input'])
                model.plot(site_img='pruned_model_2')           

                if args.symbo_regre == True:
                    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
                    model.auto_symbolic(lib=lib)
                    formula = model.symbolic_formula()[0][0]
                    print(f'After pruning and symbolic regression, the formula is obtained : \n {formula}')
            
            actual_labels_ = best_test_label
            predicted_probs_ = best_test_pred
            auc_value = calculate_auc(actual=actual_labels_, predicted=predicted_probs_)  
            AUC_list_in_cross_val.append(auc_value)   
            
            predicted_probs_10 = [1 if prob >= 0.5 else 0 for prob in predicted_probs_]
            correct_predictions = sum(1 for pred, actual in zip(predicted_probs_10, actual_labels_) if pred == actual)
            accuracy = correct_predictions / len(actual_labels_)
            # print("Accuracy:", accuracy)
            mean_accuracy += accuracy
            
            TP = sum(1 for pred, actual in zip(predicted_probs_10, actual_labels_) if pred == 1 and actual == 1)
            FN = sum(1 for pred, actual in zip(predicted_probs_10, actual_labels_) if pred == 0 and actual == 1)
            sensitivity = TP / (TP + FN)
            # print("Sensitivity:", sensitivity)
            mean_sensitivity += sensitivity
            
            TN = sum(1 for pred, actual in zip(predicted_probs_10, actual_labels_) if pred == 0 and actual == 0)
            FP = sum(1 for pred, actual in zip(predicted_probs_10, actual_labels_) if pred == 1 and actual == 0)
            specificity = TN / (TN + FP)
            # print("Specificity:", specificity)
            mean_specificity += specificity
            mean_auc += auc_value
                        
            if k_f == 0:
                tot_actual_labels_val = best_test_label
                tot_predicted_probs_val = best_test_pred
                tot_actual_labels_tra = best_train_label
                tot_predicted_probs_tra = best_train_pred
                
            elif k_f == 4 and turn_ == 0:
                # Compute calibration curve
                actual_labels_val = best_test_label
                predicted_probs_val = best_test_pred
                actual_labels_tra = best_train_label
                predicted_probs_tra = best_train_pred
                
                tot_actual_labels_val =  np.concatenate((tot_actual_labels_val, actual_labels_val))
                tot_predicted_probs_val = np.concatenate((tot_predicted_probs_val, predicted_probs_val))
                tot_actual_labels_tra = np.concatenate((tot_actual_labels_tra, actual_labels_tra))
                tot_predicted_probs_tra = np.concatenate((tot_predicted_probs_tra, predicted_probs_tra))
                
                train_fraction_of_positives, train_mean_predicted_value = continuous_calibration_curve(tot_actual_labels_tra, tot_predicted_probs_tra, n_bins=30)
                valid_fraction_of_positives, valid_mean_predicted_value = continuous_calibration_curve(tot_actual_labels_val, tot_predicted_probs_val, n_bins=30)

                # Plot calibration curve

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                ax1.plot(train_mean_predicted_value, train_fraction_of_positives, marker='o', label='Training set', markersize=1, color=(239/255, 118/255, 123/255), linewidth=5)
                #ax1.plot(valid_mean_predicted_value, valid_fraction_of_positives, marker='o', label='Validation set', markersize=1, color=(67/255, 163/255, 239/255), linewidth=5)
                ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
                ax1.set_xlabel('Predicted risk')
                ax1.set_ylabel('Observed frequency')
                ax1.set_title('a')
                #ax1.legend(frameon=False)
                ax1.grid(False)
                ax1.spines[['top', 'right']].set_visible(False)

                #ax2.plot(train_mean_predicted_value, train_fraction_of_positives, marker='o', label='Training set', markersize=1, color=(239/255, 118/255, 123/255), linewidth=5)
                ax2.plot(valid_mean_predicted_value, valid_fraction_of_positives, marker='o', label='Validation set', markersize=1, color=(67/255, 163/255, 239/255), linewidth=5)
                ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
                ax2.set_xlabel('Predicted risk')
                ax2.set_ylabel('Observed frequency')
                ax2.set_title('b')
                #ax2.legend(frameon=False)
                ax2.grid(False)
                ax2.spines[['top', 'right']].set_visible(False)

                if args.cur == True:
                    C_C_site =' Calibration_curves.png'
                    plt.savefig(C_C_site)
                    print(f"Calibration curves have been saved in {C_C_site}")
                #plt.tight_layout()
                plt.clf()
                
                def calculate_net_benefit(P, L, Pt, actual, pi):
                    A = np.sum((P >= Pt) & (actual.astype(bool)))
                    B = np.sum((P >= Pt) & ~(actual.astype(bool)))
                    return A - B * Pt / (1 - Pt)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # for train set
                p_pred=tot_predicted_probs_tra 
                actual=tot_actual_labels_tra
                p_base=tot_base_probs_tra
                pi = sum(actual) / len(actual)
                L = 0
                # print(f"pi = {pi}")
                
                Pt_values = np.linspace(0, 1, 100)
                net_benefits_pred = [calculate_net_benefit(p_pred, L, Pt, actual, pi)/len(p_pred) for Pt in Pt_values]
                ax1.plot(Pt_values, net_benefits_pred, label='Predictive Model', color=(239/255, 118/255, 123/255), linewidth=1.5)
                
                net_benefits_base = [calculate_net_benefit(p_base, L, Pt, actual, pi)/len(p_base) for Pt in Pt_values]
                ax1.plot(Pt_values, net_benefits_base, label='Base Model', color=(67/255, 163/255, 239/255), linewidth=1.5)
                
                net_benefits_none = [0 for Pt in Pt_values]
                ax1.plot(Pt_values, net_benefits_none, label='None', color=(13/255, 13/255, 13/255), linewidth=1.5)
                
                net_benefits_none = np.linspace(0, 0.5, 100)
                p_one = np.zeros_like(p_pred)
                net_benefits_all = [pi - (1-pi)*Pt / (1 - Pt) for Pt in net_benefits_none]
                ax1.plot(net_benefits_none, net_benefits_all, label='All', color=(170/255, 170/255, 170/255), linewidth=1.5)
                

                # for val set
                p_pred=tot_predicted_probs_val 
                actual=tot_actual_labels_val
                p_base=tot_base_probs_val
                pi = sum(actual) / len(actual)
                L = 0
                # print(f"pi = {pi}")
                
                Pt_values = np.linspace(0, 1, 100)
                net_benefits_pred = [calculate_net_benefit(p_pred, L, Pt, actual, pi)/len(p_pred) for Pt in Pt_values]
                ax2.plot(Pt_values, net_benefits_pred, label='Predictive Model', color=(239/255, 118/255, 123/255), linewidth=1.5)
                
                net_benefits_base = [calculate_net_benefit(p_base, L, Pt, actual, pi)/len(p_base) for Pt in Pt_values]
                ax2.plot(Pt_values, net_benefits_base, label='Base Model', color=(67/255, 163/255, 239/255), linewidth=1.5)
                
                net_benefits_none = [0 for Pt in Pt_values]
                ax2.plot(Pt_values, net_benefits_none, label='None', color=(13/255, 13/255, 13/255), linewidth=1.5)
                
                net_benefits_none = np.linspace(0, 0.5, 100)
                p_one = np.zeros_like(p_pred)
                net_benefits_all = [pi - (1-pi)*Pt / (1 - Pt) for Pt in net_benefits_none]
                ax2.plot(net_benefits_none, net_benefits_all, label='All', color=(170/255, 170/255, 170/255), linewidth=1.5)
                
                
                ax1.set_xlim(-0.04, 1.04)
                ax1.set_ylim(-0.02, 0.55)
                ax2.set_xlim(-0.04, 1.04)
                ax2.set_ylim(-0.02, 0.55)
                
                ax1.set_xlabel('High Rish Threshold')
                ax1.set_ylabel('Net Benefit')
                
                ax2.set_xlabel('High Rish Threshold')
                ax2.set_ylabel('Net Benefit')
                
                ax1.set_title('a')
                ax2.set_title('b')
                
                ax1.spines[['top', 'right']].set_visible(False)
                ax2.spines[['top', 'right']].set_visible(False)
                
                ax1.legend(frameon=False, loc='center right') 
                ax2.legend(frameon=False, loc='center right')

                if args.dca == True:
                    DCA_site =' DCA.png'
                    plt.savefig(DCA_site)
                    print(f"DCA figures have been saved in {DCA_site}")


            else :
                actual_labels_val = best_test_label
                predicted_probs_val = best_test_pred
                actual_labels_tra = best_train_label
                predicted_probs_tra = best_train_pred

                tot_actual_labels_val =  np.concatenate((tot_actual_labels_val, actual_labels_val))
                tot_predicted_probs_val = np.concatenate((tot_predicted_probs_val, predicted_probs_val))
                tot_actual_labels_tra = np.concatenate((tot_actual_labels_tra, actual_labels_tra))
                tot_predicted_probs_tra = np.concatenate((tot_predicted_probs_tra, predicted_probs_tra))
        
        avg = sum(AUC_list_in_cross_val) / 5
        AUC_list_per_cross_val.append(avg)


    auc_mean_final, auc_ci = auc_mean_ci(AUC_list_per_cross_val)
    
    separator = '-' * 50


    mean_auc = mean_auc / args.N / 5
    mean_accuracy = mean_accuracy / args.N / 5
    mean_sensitivity = mean_sensitivity / args.N / 5
    mean_specificity = mean_specificity / args.N / 5


    print(separator.center(50))
    print("Cross-Validation AUC".center(50))
    print(separator.center(50))
    # print(f"\nAUC in Cross-Validation: {AUC_list_per_cross_val}\n")
    print(f"AUC = {mean_auc:.3f}".center(50))
    print(f"95% CI: ({auc_ci[0]:.3f}, {auc_ci[1]:.3f})".center(50))
    print(f"Accuracy = {mean_accuracy:.3f}".center(50))
    print(f"Sensitivity = {mean_sensitivity:.3f}".center(50))
    print(f"Specificity = {mean_specificity:.3f}\n".center(50))


    mean_nri, ci, p_value = get_ci_p(value_list=nri_train_list)
    print(separator.center(50))
    print("NRI for Train Set".center(50))
    print(separator.center(50))
    print(f"Aver NRI: {mean_nri:.3f}".center(50))
    print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})".center(50))
    print(f"p-value: {p_value:.6f}\n".center(50))


    mean_nri, ci, p_value = get_ci_p(value_list=nri_test_list)
    print(separator.center(50))
    print("NRI for Validation Set".center(50))
    print(separator.center(50))
    print(f"Aver NRI: {mean_nri:.3f}".center(50))
    print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})".center(50))
    print(f"p-value: {p_value:.6f}\n".center(50))


    mean_nri, ci, p_value = get_ci_p(value_list=idi_train_list)
    print(separator.center(50))
    print("IDI for Train Set".center(50))
    print(separator.center(50))
    print(f"Aver IDI: {mean_nri:.3f}".center(50))
    print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})".center(50))
    print(f"p-value: {p_value:.6f}\n".center(50))


    mean_nri, ci, p_value = get_ci_p(value_list=idi_test_list)
    print(separator.center(50))
    print("IDI for Validation Set".center(50))
    print(separator.center(50))
    print(f"Aver IDI: {mean_nri:.3f}".center(50))
    print(f"95% CI: ({ci[0]:.3f}, {ci[1]:.3f})".center(50))
    print(f"p-value: {p_value:.6f}\n".center(50))

    


if __name__ == "__main__":
    evaluate()