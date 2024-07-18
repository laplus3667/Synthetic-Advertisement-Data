#%%
import os
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from scipy.stats import wasserstein_distance

# 創建一個字典來存儲你想要訓練的模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    # 'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'GaussianNB': GaussianNB(),
    'ComplementNB': ComplementNB(),
    'MultinomialNB': MultinomialNB(),
    'DNN': None,
    'DeepFM': None
}

data = pd.read_csv('../../train_data_ads.csv')
print("data is loaded...")

#%%
class UtilityEvaluator:
    def __init__(self, task_id, label_rate, synthetic_path, predictions_path=None):
          self.task_id = task_id
          self.label_rate = label_rate
          self.synthetic_path = synthetic_path
          self.predictions_path = predictions_path
    
    def calculate_label_rate(self, df):
        # Get the total number of samples
        total_samples = len(df)
        
        # Count the occurrences of each label in the 'label' column of df dataframe.
        label_counts = df['label'].value_counts()
        # Extract the count of positive labels (label == 1) from the label_counts series.
        positive_count = label_counts.get(1, 0)
        # Extract the count of negative labels (label == 0) from the label_counts series.
        negative_count = label_counts.get(0, 0)
        # Calculate the rate of positive labels to negative labels.
        # label_rate = positive_count / negative_count if negative_count != 0 else 0
        label_rate = positive_count / total_samples
        
        # Print the sizes of positive and negative samples along with the label rate, formatted to two decimal places.
        print("Total Sample size is {}, Positive Sample size is {}, Negative Sample size is {}, label rate is {:.4f}".format(total_samples, positive_count, negative_count, label_rate))
    
    def split_data(self):
        """
        get the train, holdout, valid datasets by splitting the task data
        """
        # get the task data
        df_task = data[data['task_id'] == self.task_id]
        print(f"Task data shape is {df_task.shape}")
        columns_to_drop = [column for column in df_task.columns if df_task[column].nunique() > 1000 or df_task[column].nunique() == 1]
        df_task = df_task.drop(columns=columns_to_drop)
        df_task = df_task.select_dtypes(include=[np.number])
        self.df_task = df_task

        df_label_0 = self.df_task[self.df_task['label'] == 0]
        df_label_1 = self.df_task[self.df_task['label'] == 1]

        # 對 label 為 0 的數據集進行分割
        total_samples_label_0 = len(df_label_0)
        train_size_label_0 = int(0.4 * total_samples_label_0)
        holdout_size_label_0 = int(0.4 * total_samples_label_0)
        # 對 label 為 1 的數據集進行分割
        total_samples_label_1 = len(df_label_1)
        train_size_label_1 = int(0.4 * total_samples_label_1)
        holdout_size_label_1 = int(0.4 * total_samples_label_1)

        df_label_0_train = df_label_0.sample(n=train_size_label_0, random_state=42)
        df_label_0_hold = df_label_0.drop(df_label_0_train.index).sample(n=holdout_size_label_0, random_state=42)
        df_label_0_val = df_label_0.drop(df_label_0_train.index).drop(df_label_0_hold.index)

        df_label_1_train = df_label_1.sample(n=train_size_label_1, random_state=42)
        df_label_1_hold = df_label_1.drop(df_label_1_train.index).sample(n=holdout_size_label_1, random_state=42)
        df_label_1_val = df_label_1.drop(df_label_1_train.index).drop(df_label_1_hold.index)
        
        # 將 label 為 0 和 label 為 1 的 train 子數據集合併成 df_train
        df_train = pd.concat([df_label_0_train, df_label_1_train])
        # 將 label 為 0 和 label 為 1 的 holdout 子數據集合併成 df_holdout
        df_holdout = pd.concat([df_label_0_hold, df_label_1_hold])
        # 將 label 為 0 和 label 為 1 的 holdout 子數據集合併成 df_val
        df_val = pd.concat([df_label_0_val, df_label_1_val])

        self.calculate_label_rate(self.df_task)
        self.calculate_label_rate(df_train)
        self.calculate_label_rate(df_holdout)
        self.calculate_label_rate(df_val)

        print(f"Train data shape is {df_train.shape}")
        print(f"Holdout data shape is {df_holdout.shape}")
        print(f"Validation data shape is {df_val.shape}")

        # shuffle the datasets
        self.df_train = df_train.sample(frac=1, random_state = 42).reset_index(drop=True)
        self.df_holdout = df_holdout.sample(frac=1, random_state = 42).reset_index(drop=True)
        self.df_val = df_val.sample(frac=1, random_state = 42).reset_index(drop=True)

        # 儲存隨機排序後的df_train
        self.df_train.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_train.csv', index=False)
        df_label_0_train = df_label_0_train.drop("label", axis=1)
        df_label_1_train = df_label_1_train.drop("label", axis=1)
        print(f"Train0 data shape is {df_label_0_train.shape}")
        print(f"Train1 data shape is {df_label_1_train.shape}")
        df_label_0_train.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_train0.csv', index=False)
        df_label_1_train.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_train1.csv', index=False)

        # 儲存隨機排序後的df_holdout
        self.df_holdout.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_holdout.csv', index=False)
        # 儲存隨機排序後的df_val和df_test
        self.df_val.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_val.csv', index=False)
        # self.df_test.to_csv(f'../../data/{self.task_id}/df_{self.task_id}_test.csv', index=False)

    def merge_synth_data(self):
        """
        After generating the synthetic data with both label 1 and 0, merge them to get the synthetic_train data and save to a csv file
        """
        base_path = f'../../data/{self.task_id}'
        df_synthetic_train0 = pd.read_csv(f'{base_path}/{self.synthetic_path}/synthetic_df_{self.task_id}_train0.csv')
        df_synthetic_train1 = pd.read_csv(f'{base_path}/{self.synthetic_path}/synthetic_df_{self.task_id}_train1.csv')
        df_synthetic_train0['label'] = 0
        df_synthetic_train1['label'] = 1
        df_synthetic_train  = pd.concat([df_synthetic_train0, df_synthetic_train1])
        df_synthetic_train = df_synthetic_train.sample(frac=1, random_state=42).reset_index(drop=True)
        synthetic_train_path = f'{base_path}/{self.synthetic_path}/synthetic_df_{self.task_id}_train.csv'
        df_synthetic_train.to_csv(synthetic_train_path, index=False)

    def read_data(self):
        """
        if all the files exist already, just read the datasets
        """
        base_path = f'../../data/{self.task_id}'
        self.df_train = pd.read_csv(f'{base_path}/df_{self.task_id}_train.csv')
        self.df_holdout = pd.read_csv(f'{base_path}/df_{self.task_id}_holdout.csv')
        self.df_val = pd.read_csv(f'{base_path}/df_{self.task_id}_val.csv')
        self.df_synthetic_train = pd.read_csv(f'{base_path}/{self.synthetic_path}/synthetic_df_{self.task_id}_train.csv')
        
        print("The train data: ")
        self.calculate_label_rate(self.df_train)
        print("The holdout data: ")
        self.calculate_label_rate(self.df_holdout)
        print("The validation data: ")
        self.calculate_label_rate(self.df_val)
        print("The synthetic train data: ")
        self.calculate_label_rate(self.df_synthetic_train)

    def prep_data_modeling(self):
        # 假設你的目標變量列名叫 'label'
        self.X_train = self.df_train.drop('label', axis=1)  # 訓練集特徵數據
        self.y_train = self.df_train['label']  # 訓練集目標數據

        self.X_holdout = self.df_holdout.drop('label', axis=1)  # holdout集特徵數據
        self.y_holdout = self.df_holdout['label']  # holdout集目標數據

        if self.df_synthetic_train is not None:
            self.X_synthetic_train = self.df_synthetic_train.drop('label', axis=1)  # synthetic data based on training data
            self.y_synthetic_train = self.df_synthetic_train['label']  # synthetic data based on training data

        self.X_val = self.df_val.drop('label', axis=1)  # 驗證集特徵數據
        self.y_val = self.df_val['label']  # 驗證集目標數據

    # Function to convert DataFrames to JSON-compatible dictionaries recursively
    def dataframe_to_dict(self, df):
        return df.to_dict(orient='split')

    def get_results_and_plot(self):
        datasets = {"train": (self.X_train, self.y_train), 
            "holdout": (self.X_holdout, self.y_holdout), 
            "synthetic": (self.X_synthetic_train, self.y_synthetic_train)}
        
        # Initialize subplot indices
        roc_subplot_index = 1
        pr_subplot_index = 2

        # train_tpr_scores = {}
        # holdout_tpr_scores = {}
        # synthetic_tpr_scores = {}

        results = pd.DataFrame(columns=['Accuracy', 'AUC', 'TPR', 'FPR', 'TNR', 'FNR'])

        for data in datasets:

            # 設定圖片尺寸和布局
            plt.figure(figsize=(12, 6))

            # 第一個subplot為ROC曲線
            plt.subplot(1, 2, 1)
            colors = [
                'blue',      # Logistic Regression
                'green',     # Decision Tree (tree-based)
                'lightgreen',# Random Forest (tree-based)
                'darkgreen', # Gradient Boosting (tree-based)
                'yellowgreen',# XGBoost (tree-based)
                'red',       # GaussianNB (NB model)
                'lightcoral',# ComplementNB (NB model)
                'indianred', # MultinomialNB (NB model)
                'purple',    # DNN (deep learning model)
                'mediumpurple' # DeepFM (deep learning model)
            ]
            model_index = 0

            # 第二個subplot為Precision-Recall曲線
            plt.subplot(1, 2, 2)
            colors = ['blue', 'green', 'lightgreen', 'darkgreen', 'yellowgreen', 'red', 'lightcoral', 'indianred', 'purple', 'mediumpurple']  # 重設顏色索引
            
            for name, model in models.items():
                curr_data = f"{data}_{name}"
                # results[current_row_value] = curr_data

                if name == 'DNN' or name == 'DeepFM':
                    y_proba = np.loadtxt(f"../../data/{self.task_id}/predictions/predictions_original/{name}_{data}_{self.task_id}_predictions.txt")
                    if data == "synthetic":
                        y_proba = np.loadtxt(f"../../data/{self.task_id}/predictions/{self.predictions_path}/{name}_{data}_{self.task_id}_predictions.txt")
                    y_pred = (y_proba > 0.5).astype(int)
                else:
                    try:
                        model.fit(datasets[data][0], datasets[data][1])
                    except Exception as e:
                        print(data, name, e)
                        break
                    y_pred = model.predict(self.X_val)
                    y_proba = model.predict_proba(self.X_val)[:, 1]  # 獲得正類別的預測概率

                # 計算準確度和AUC
                accuracy = accuracy_score(self.y_val, y_pred)
                auc_score = roc_auc_score(self.y_val, y_proba)

                # 計算混淆矩陣並抽取TP, FN, FP, TN
                tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()

                # 計算性能指標
                tpr = tp / (tp + fn)  # 真陽性率
                fpr = fp / (fp + tn)  # 假陽性率
                tnr = tn / (tn + fp)  # 真陰性率
                fnr = fn / (tp + fn)  # 假陰性率

                # 固定小數點後三位並儲存結果
                results.loc[curr_data] = {
                    'Accuracy': round(accuracy, 3),
                    'AUC': round(auc_score, 3),
                    'TPR': round(tpr, 3),
                    'FPR': round(fpr, 3),
                    'TNR': round(tnr, 3),
                    'FNR': round(fnr, 3)
                }
                # results.loc[curr_data, 'Accuracy'] = round(accuracy, 3)
                # results.loc[curr_data, 'AUC'] = round(auc_score, 3)
                # results.loc[curr_data, 'TPR'] = round(tpr, 3)
                # results.loc[curr_data, 'FPR'] = round(fpr, 3)
                # results.loc[curr_data, 'TNR'] = round(tnr, 3)
                # results.loc[curr_data, 'FNR'] = round(fnr, 3)
                # print(results)
                
                # 計算ROC曲線及AUC
                fpr, tpr, _ = roc_curve(self.y_val, y_proba)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                plt.subplot(1, 2, roc_subplot_index)
                fpr, tpr, _ = roc_curve(self.y_val, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=colors[model_index], lw=2, label=f'{name} ROC (area = {roc_auc:.2f})')

                # Plot Precision-Recall curve
                plt.subplot(1, 2, pr_subplot_index)
                precision, recall, _ = precision_recall_curve(self.y_val, y_proba)
                ap = average_precision_score(self.y_val, y_proba)
                plt.plot(recall, precision, color=colors[model_index], lw=2, label=f'{name} PR (AP = {ap:.2f})')

                model_index += 1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc="upper right")

            plt.tight_layout()
            plt.show()
        
        results.to_csv(f"../pipeline_results/utility/{self.task_id}/{self.task_id}_mixed_{self.label_rate}.csv", index=True)

        # train_tpr_df = pd.DataFrame(list(synthetic_tpr_scores.items()), columns=['Model', 'TPR'])
        # train_tpr_df.to_csv(f"../pipeline_results/utility/{self.task_id}/{self.task_id}_train.csv", index = False)
        # holdout_tpr_df = pd.DataFrame(list(synthetic_tpr_scores.items()), columns=['Model', 'TPR'])
        # holdout_tpr_df.to_csv(f"../pipeline_results/utility/{self.task_id}/{self.task_id}_holdout.csv", index = False)

        # synthetic_tpr_df = pd.DataFrame(list(synthetic_tpr_scores.items()), columns=['Model', 'TPR'])
        # synthetic_tpr_df.to_csv(f"../pipeline_results/utility/{self.task_id}/{self.task_id}_{self.synthetic_path}.csv", index = False)
        # print("tpr saved!")

    def process_evaluation(self):
        self.read_data()
        self.prep_data_modeling()
        self.get_results_and_plot()

#%%
label_rate = "0.04"
evaluator = UtilityEvaluator(
    task_id = 31941,
    label_rate = label_rate,
    synthetic_path = f"synthetic_{label_rate}", # "synthetic_original" or "synthetic_{label_rate}"
    predictions_path = f"predictions_{label_rate}" # "predictions_original" or "predictions_{label_rate}"
)
# evaluator.split_data() # if we run this task data for the first time
# evaluator.merge_synth_data() # merge the synthetic label 1 and 0 data
evaluator.process_evaluation() # if all the data files are ready
# %%
