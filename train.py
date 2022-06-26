#!env python38
# -*- coding：utf-8 -*-
# @Date     : 2022/6/9 17:06
# @Author   : John_pep
# @Description:
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import os
import torch
import sklearn
import scipy

from datetime import datetime
from tqdm import tqdm
from scipy import stats
from platform import python_version

import utils_bsc
def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Device: GPU =', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Device: CPU')

    saved_results = 'training_results'

    # 检查结果文件必须保存的目录是否存在
    if not os.path.exists(saved_results):
        # 创建一个新目录，因为它不存在
        os.makedirs(saved_results)

    # 这个单元格是在 google colab 中使用这个笔记本所必需的
    # 如果在 colab 中运行此笔记本，请将 colab 更改为 True

    colab = False

    if colab is True:
        cwd = os.getcwd()
        # if cwd != "/content/Bsc_Thesis":
        #     ! git clone https: // github.com / SergioTallo / Bsc_Thesis.git
        #     % cd  Bsc_Thesis

        print(cwd)

    dataset = pd.read_csv('data_factory.csv')
    dataset.head()
    # 用 NaN 替换所有缺失值
    dataset = dataset.replace(' ', np.nan)
    # 搜索所有具有 NaN 值的行
    nan_values = dataset[dataset.isna().any(axis=1)]
    # 打印形状以知道有多少
    print(f'Number of rows with NaN values before cleaning: {nan_values.shape[0]}')

    # 用前一行值填充所有 NaN 值
    dataset_clean = dataset.fillna(method='ffill')

    # 检查是否没有任何 NaN 值
    nan_values = dataset_clean[dataset_clean.isna().any(axis=1)]
    # 打印形状以知道有多少
    print(f'Number of rows with NaN values after cleaning: {nan_values.shape[0]}')

    # 样本总数
    print(f'Total number of samples: {dataset_clean.shape[0]}')
    print(f'Number of features: {dataset_clean.shape[1]}')
    print_data = True
    print_graphs = False
    if print_data is True:
        for column in dataset_clean.columns:
            if column == 'time':
                print(column)
                print('Min value: ', dataset_clean[column].min())
                print('Max value: ', dataset_clean[column].max())
                print('')
            else:
                print(column)
                print('Min value: ', dataset_clean[column].min())
                print('Max value: ', dataset_clean[column].max())
                print('Mean value: ', dataset_clean[column].mean())
                print('Median value: ', dataset_clean[column].median())
                print('Standard deviation: ', dataset_clean[column].std())
                print('')

    if print_graphs is True:

        for i, column in enumerate(dataset_clean.columns):
            if i > 0:
                # 每周间隔的特征
                # Feature in a weekly interval
                utils_bsc.week_plot(dataset_clean, i, column)
                # Feature in a daily interval (only the values of weekdays between 4:00 and 19:30)
                # 每日间隔中的特征（仅限工作日 4:00 到 19:30 之间的值）
                utils_bsc.daily_plot(dataset_clean, i, column)
    # 打印一些图表来显示每个特征的密度分布
    if print_graphs is True:
        for column in tqdm(dataset_clean.columns):
            if column != 'time':
                sns.displot(dataset_clean, x=column, kind="kde")

    # 创建了两个额外的数据集，一个包含工作日 4:00 到 19:30 之间，一个包含其余数据集。
    dataset_clean_time = pd.to_datetime(dataset_clean['time'])

    day_mask = dataset_clean_time.dt.day_name()

    time_mask = (dataset_clean_time.dt.hour >= 4) & ((dataset_clean_time.dt.hour < 19) | (
                (dataset_clean_time.dt.hour == 19) & (dataset_clean_time.dt.minute <= 30))) & (
                            (day_mask == ('Monday')) | (day_mask == ('Tuesday')) | (day_mask == ('Wednesday')) | (
                                day_mask == ('Thursday')) | (day_mask == ('Friday')))

    dataset_weekdays = dataset_clean[time_mask]

    for i in range(len(time_mask)):
        if time_mask[i] == False:
            time_mask[i] = True
        elif time_mask[i] == True:
            time_mask[i] = False

    dataset_weekend = dataset_clean[time_mask]

    print(f'Weekdays dataset size: {len(dataset_weekdays)}')
    print(f'Weekend dataset size: {len(dataset_weekend)}')
    if print_graphs is True:
        for column in tqdm(dataset_weekdays.columns):
            if column != 'time':
                sns.displot(dataset_weekdays, x=column, kind="kde")
    if print_graphs is True:
        for column in tqdm(dataset_weekend.columns):
            if column != 'time':
                sns.displot(dataset_weekend, x=column, kind="kde")
    # 在整个数据集中执行数据规范化。 如果需要，我们可以打印数据的分布。
    dataset_norm = utils_bsc.normalize_mean_std_dataset(dataset_clean)

    print_graphs = False

    if print_graphs is True:
        for column in tqdm(dataset_norm.columns):
            if column != 'time':
                sns.displot(dataset_norm, x=column, kind="kde")
    # 在工作日数据集中执行数据规范化。 如果需要，我们可以打印数据的分布。
    dataset_weekdays_norm = utils_bsc.normalize_mean_std_dataset(dataset_weekdays)

    print_graphs = False

    if print_graphs is True:
        for column in tqdm(dataset_weekdays_norm.columns):
            if column != 'time':
                sns.displot(dataset_weekdays_norm, x=column, kind="kde")

    # P在工作日数据集中执行数据规范化。 如果需要，我们可以打印数据的分布。
    dataset_weekend_norm = utils_bsc.normalize_mean_std_dataset(dataset_weekend)

    print_graphs = False

    if print_graphs is True:
        for column in tqdm(dataset_weekend_norm.columns):
            if column != 'time':
                sns.displot(dataset_weekend_norm, x=column, kind="kde")

    correlations = []
    matrix = []

    for i in dataset_norm.columns[1:]:
        feature = []
        for j in dataset_norm.columns[1:]:
            print(f'Correlation between {i} and {j}')
            correlation = stats.pearsonr(dataset_norm[i], dataset_norm[j])[0]
            if i != j:
                correlations.append(abs(correlation))
                feature.append(abs(correlation))
                print(correlation)
        print(f'Mean of {i} correlations: {np.mean(feature)}')
        print('')
        matrix.append(feature)

    print(f'Mean of all correlations: {np.mean(correlations)}')
    # 特征相关性热图

    corr = dataset_norm.corr()
    sns.heatmap(corr, cmap="Blues")

    # 协方差矩阵、特征值和解释方差

    covmatrix = dataset_norm.cov()
    eigenvalues, eigenvectors = np.linalg.eig(covmatrix)

    acc = 0

    acc_variance = []

    for i, eigen in enumerate(eigenvalues):
        acc += eigen / np.sum(eigenvalues)
        acc_variance.append(acc)
        print(
            f'Explained_variance {i + 1} principal component: {eigen / np.sum(eigenvalues)} (accumulated {round(acc, 4)})')

    fig = plt.figure(figsize=(15, 8))

    a = acc_variance

    b = [i + 1 for i in range(len(acc_variance))]
    plt.title('Explained variance over number of principal components')
    plt.xlabel('n principal components')
    plt.xticks(b)
    plt.ylabel('explained variance')
    plt.bar(b, a)
    plt.show()

    loader_train, loader_test = utils_bsc.create_dataloaders(dataset=dataset_norm, device=device)
    criterion = nn.MSELoss()

    losses_train = []

    for i in loader_train:
        output = i[0]
        target = i[1]
        loss = criterion(output, target)
        losses_train.append(loss.item())

    losses_test = []

    for i in loader_test:
        output = i[0]
        target = i[1]
        loss = criterion(output, target)
        losses_test.append(loss.item())

    # 保存到 npy 文件以跟踪结果并打印图表
    np.save(saved_results + '/baseline_train.npy', losses_train)
    np.save(saved_results + '/baseline_test.npy', losses_test)

    # if colab is True:
    # files.download(saved_results + '/baseline_train.npy')
    # files.download(saved_results + '/baseline_test.npy')
    print("Training set")
    print("Mean Loss of baselinemodel: ", np.mean(losses_train))
    print("Standard deviation Loss of baselinemodel: ", np.std(losses_train))
    print('\n')
    print("Test set")
    print("Mean Loss of baselinemodel: ", np.mean(losses_test))
    print("Standard deviation Loss of baselinemodel: ", np.std(losses_test))

    start_train_FFN = True

    # 创建模型 FFN 实例
    model_FFN = utils_bsc.ANN_relu(18, 18).to(device)

    print(f'Model: {type(model_FFN).__name__}')
    print(f'{utils_bsc.count_parameters(model_FFN)} trainable parameters.')

    # 定义损失
    criterion = nn.MSELoss()

    # 定义优化器
    learning_rate = 0.01
    optimizer_whole = torch.optim.SGD(model_FFN.parameters(), lr=learning_rate)

    if start_train_FFN is True:
        n_epochs = 1

        params_not_trained_whole = model_FFN.parameters()

        start_time = datetime.now()

        global test_losses_FFN

        best_results, train_losses_FFN, test_losses_FFN = utils_bsc.train_FFN(model_FFN, criterion, optimizer_whole,
                                                                              loader_train, loader_test, n_epochs)
        print(test_losses_FFN)

        model_FFN = best_results[0]
        best_train_loss = best_results[1]
        best_test_loss = best_results[2]
        best_epoch_number = best_results[3]

        end_time = datetime.now()
        time_diff = (end_time - start_time)
        execution_time = time_diff.total_seconds()

        print(f'Best test loss at epoch {best_epoch_number}')
        print(f'Train Loss: {best_train_loss}')
        print(f'Test Loss: {best_test_loss}')
        print(f'\nTraining time for {n_epochs} epochs: {execution_time} seconds')

        # save to npy file
        np.save(saved_results + '/FFN_train.npy', train_losses_FFN)
        np.save(saved_results + '/FFN_test.npy', test_losses_FFN)
        torch.save(model_FFN, saved_results + '/model_FFN.pt')

        # if colab is True:
        #     # files.download(saved_results + '/FFN_train.npy')
        #     # files.download(saved_results + '/FFN_test.npy')
        #     # files.download(saved_results + '/model_FFN.pt')

    if start_train_FFN is True:
        baseline_loss = [np.mean(losses_train) for i in range(len(train_losses_FFN))]
        utils_bsc.print_results_training(train_loss=train_losses_FFN, test_loss=test_losses_FFN,
                                         test_loss_baseline=baseline_loss, baseline_label='Baseline',
                                         title="Full Forward Neural Network train results")

    training_results_transformers = {}

    models = {'vanilla': [6, 1, 6, 2048, 'SGD', 0.01, None, True, 30, 16]}
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers,
                                                                        saved_results, colab)
    if models['vanilla'][7] is True:
        baseline_loss = [np.mean(i) for i in test_losses_FFN]
        utils_bsc.print_results_training(train_loss=training_results_transformers['vanilla'][4],
                                         test_loss=training_results_transformers['vanilla'][5],
                                         test_loss_baseline=baseline_loss, baseline_label='FFN Test Loss',
                                         title="Training results " + 'vanilla' + " Transformer (" + str(
                                             models['vanilla'][0]) + " encoder layers, " + str(
                                             models['vanilla'][1]) + " decoder layer, " + str(
                                             models['vanilla'][2]) + " heads. " + models['vanilla'][4])
    models['vanilla'][7] = False
    models['ADAM'] = [6, 1, 6, 2048, 'ADAM', 0.001, None, True, 30, 16]
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers, saved_results,
                                                                        colab)
    if models['ADAM'][7] is True:
        baseline_loss = [np.mean(i) for i in training_results_transformers['vanilla'][5]]
        utils_bsc.print_results_training(train_loss=training_results_transformers['ADAM'][4],
                                         test_loss=training_results_transformers['ADAM'][5],
                                         test_loss_baseline=baseline_loss,
                                         baseline_label='Vanilla transformer Test Loss',
                                         title="Training results " + 'ADAM' + " Transformer (" + str(
                                             models['ADAM'][0]) + " encoder layers, " + str(
                                             models['ADAM'][1]) + " decoder layer, " + str(
                                             models['ADAM'][2]) + " heads. " + models['ADAM'][4])
    models['ADAM'][7] = False
    models['Momentum'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 30, 16]
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers, saved_results,
                                                                        colab)
    if models['Momentum'][7] is True:
        baseline_loss = [np.mean(i) for i in training_results_transformers['vanilla'][5]]
        utils_bsc.print_results_training(train_loss=training_results_transformers['Momentum'][4],
                                         test_loss=training_results_transformers['Momentum'][5],
                                         test_loss_baseline=baseline_loss,
                                         baseline_label='Vanilla transformer Test Loss',
                                         title="Training results " + 'Momentum' + " Transformer (" + str(
                                             models['Momentum'][0]) + " encoder layers, " + str(
                                             models['Momentum'][1]) + " decoder layer, " + str(
                                             models['Momentum'][2]) + " heads. " + models['Momentum'][4])
    models['Momentum'][7] = False
    models['smallest'] = [1, 1, 1, 512, 'SGD', 0.001, 0.9, True, 30, 16]
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers, saved_results,
                                                                        colab)
    if models['smallest'][7] is True:
        baseline_loss = [np.mean(i) for i in training_results_transformers['vanilla'][5]]
        utils_bsc.print_results_training(train_loss=training_results_transformers['smallest'][4],
                                         test_loss=training_results_transformers['smallest'][5],
                                         test_loss_baseline=baseline_loss,
                                         baseline_label='Vanilla transformer Test Loss',
                                         title="Training results " + 'smallest' + " Transformer (" + str(
                                             models['smallest'][0]) + " encoder layers, " + str(
                                             models['smallest'][1]) + " decoder layer, " + str(
                                             models['smallest'][2]) + " heads. " + models['smallest'][4])
    models['smallest'][7] = False
    models['bigger'] = [10, 5, 9, 4096, 'SGD', 0.001, 0.9, True, 30, 16]
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers, saved_results,
                                                                        colab)
    if models['bigger'][7] is True:
        baseline_loss = [np.mean(i) for i in training_results_transformers['vanilla'][5]]
        utils_bsc.print_results_training(train_loss=training_results_transformers['bigger'][4],
                                         test_loss=training_results_transformers['bigger'][5],
                                         test_loss_baseline=baseline_loss,
                                         baseline_label='Vanilla transformer Test Loss',
                                         title="Training results " + 'bigger' + " Transformer (" + str(
                                             models['bigger'][0]) + " encoder layers, " + str(
                                             models['bigger'][1]) + " decoder layer, " + str(
                                             models['bigger'][2]) + " heads. " + models['bigger'][4])
    models['bigger'][7] = False
    models['seq_15'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 15, 16]
    models['seq_60'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 60, 16]
    models['seq_2'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 2, 16]
    models['seq_20'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 20, 16]
    models['seq_10'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 10, 16]
    models['seq_120'] = [6, 1, 6, 2048, 'SGD', 0.001, 0.9, True, 120, 16]
    training_results_transformers = utils_bsc.define_train_transformers(models, device, dataset_norm,
                                                                        training_results_transformers, saved_results,
                                                                        colab)

    class LSTM(torch.nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int, bias_init: float):
            super(LSTM, self).__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(self.lstm.hidden_size, output_size)

            # Deactivate forget gate to be in line with the original definition.
            def _reset_forget_gate_hook(_gradients: torch.Tensor) -> torch.Tensor:
                _gradients[_gradients.shape[0] // 4:_gradients.shape[0] // 2].fill_(0.0)
                return _gradients

            for name, parameter in self.lstm.named_parameters():
                if 'bias' in name:
                    parameter.data[(parameter.shape[0] // 4):(parameter.shape[0] // 2)].fill_(bias_init)
                    parameter.register_hook(_reset_forget_gate_hook)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.lstm(x)[0][:, -1, :]
            return self.fc(x)

    def training_lstm(model, optimizer, criterion, train_loader, test_loader, n_epochs, train_loss=None,
                             test_loss=None):
        if train_loss is not None:
            epoch_loss_train = train_loss
            best_train_loss = min([np.mean(i) for i in train_loss])
            best_epoch = np.where(min([np.mean(i) for i in test_loss]))
        else:
            epoch_loss_train = []
            best_train_loss = 9999999999
            best_epoch = 0

        if test_loss is not None:
            epoch_loss_test = test_loss
            best_test_loss = min([np.mean(i) for i in test_loss])
            best_epoch = np.where(min([np.mean(i) for i in train_loss]))
        else:
            epoch_loss_test = []
            best_test_loss = 99999999999
            best_epoch = 0

        best_model = model
        starting_epoch = len(epoch_loss_test)

        for e in range(1, n_epochs + 1):

            print(f'Epoch: {e + starting_epoch} of {n_epochs}')
            print('Training...')
            model.train()

            for i in tqdm(train_loader):
                input = i[0]
                target = i[1]

                net_out = model.forward(input)

                # Compute loss
                loss = criterion(net_out, target)

                optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Optimization
                optimizer.step()

            print('\nTest with training set')
            losses_train = []
            model.eval()
            with torch.no_grad():
                for i in tqdm(train_loader):
                    input = i[0]
                    target = i[2]

                    net_out = model.forward(input)

                    # Compute loss
                    losses_train.append(float(criterion(net_out, target).item()))

            print('\nCurrent Mean loss Train Set: ', np.mean(losses_train))
            epoch_loss_train.append(losses_train)

            print('\nTest with test set')
            losses_test = []
            model.eval()

            with torch.no_grad():
                for i in tqdm(test_loader):
                    input = i[0]
                    target = i[1]

                    net_out = model.forward(input)

                    # Compute loss
                    losses_test.append(float(criterion(net_out, target).item()))

            print('\nCurrent Mean loss Test Set: ', np.mean(losses_test))
            epoch_loss_test.append(losses_test)

            print('\n')

            if np.mean(losses_test) < best_test_loss:
                best_test_loss = np.mean(losses_test)
                best_train_loss = np.mean(losses_train)
                best_model = model
                best_epoch = e

        return (best_model, best_train_loss, best_test_loss, best_epoch), epoch_loss_train, epoch_loss_test

    def define_train_lstm(models, device, dataset, training_results_transformers, path_save, colab):
        for i in models:
            if models[i][7] is True:

                loader_train, loader_test = utils_bsc.create_sequece_dataloaders(dataset=dataset, seq_length=models[i][8],
                                                                       batch_size=models[i][9], device=device)

                target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = LSTM(input_size=18, hidden_size= 32, output_size=18, bias_init=0.0).to(target_device)

                print(f'Model: {type(model).__name__} - {i}')
                print(f'{utils_bsc.count_parameters(model)} trainable parameters.')

                n_epochs = 1
                learning_rate = 0.01

                if models[i][4] == 'SGD':

                    if models[i][6] is not None:
                        optimizer = torch.optim.SGD(model.parameters(), lr=models[i][5], momentum=models[i][6])
                    else:
                        optimizer = torch.optim.SGD(model.parameters(), lr=models[i][5])
                elif models[i][4] == 'ADAM':
                    optimizer = torch.optim.Adam(model.parameters(), lr=models[i][5])

                criterion = nn.MSELoss()

                start_time = datetime.now()

                best_results, train_losses, test_losses = training_lstm(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=loader_train,
                    test_loader=loader_test,
                    n_epochs=n_epochs)

                Transformer_trained_Model = best_results[0]
                best_train_loss = best_results[1]
                best_test_loss = best_results[2]
                best_epoch_number = best_results[3]

                end_time = datetime.now()
                time_diff = (end_time - start_time)
                execution_time = time_diff.total_seconds()

                print(f'Best test loss at epoch {best_epoch_number}')
                print(f'Train Loss: {best_train_loss}')
                print(f'Test Loss: {best_test_loss}')
                print(f'\nTraining time for {n_epochs} epochs: {execution_time} seconds')

                print(f'Training time: {execution_time} seconds')

                training_results_transformers[i] = [Transformer_trained_Model, best_train_loss, best_test_loss,
                                                    best_epoch_number, train_losses, test_losses, execution_time]

                # save to npy file
                np.save(path_save + '/Transformer_' + i + '_train.npy', train_losses)
                np.save(path_save + '/Transformer_' + i + '_test.npy', test_losses)
                torch.save(Transformer_trained_Model, path_save + '/Transformer_' + models[i][8] + '.pt')

                if colab is True:
                    from google.colab import files

                    files.download(path_save + '/Transformer_' + i + '_train.npy')
                    files.download(path_save + '/Transformer_' + i + '_test.npy')
                    files.download(path_save + '/Transformer_' + i + '.pt')

        return training_results_transformers
    #%%
    models = {}
    training_results_lstm = {}

    models['lstm'] = [1, 1, 1, 512, 'SGD', 0.001, 0.9, True, 30, 16]

    training_results_lstm = define_train_lstm(models, device, dataset_norm, training_results_lstm, saved_results, colab)

if __name__=='__main__':
    main()