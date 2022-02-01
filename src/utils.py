import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Split dataframe into trainset and testset
def train_test_split(data, test_split = 0.15):
    idx = np.random.permutation(np.arange(len(data)))
    split_idx = int(len(idx) * (1 - test_split))
    return data.iloc[idx[:split_idx], :].reset_index(drop=True), data.iloc[idx[split_idx:], :].reset_index(drop=True)

# Cross validation
def cross_validation(model, data, fold = 5, score = 'acc'):
    results = []
    for i in range(fold):
        # Split train dataset into train set, validation set
        train_df, val_df = train_test_split(data, 1 / fold)
        X_train = train_df.iloc[:, :-1].to_numpy()
        y_train = train_df.iloc[:, -1].to_numpy()
        X_val = val_df.iloc[:, :-1].to_numpy()
        y_val = val_df.iloc[:, -1].to_numpy()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate accuracy
        res = get_classification_results(y_pred, y_val)
        # print(res)
        results.append(res[score])

    return results

# Accuracy, precision, recall, f1 score
def get_classification_results(pred, true):
    # True positives
    tp = np.sum(np.where((pred == true) & (true == 1), 1, 0))
    # True negatives
    tn = np.sum(np.where((pred == true) & (true == 0), 1, 0))
    # False positives
    fp = np.sum(np.where((pred != true) & (pred == 1), 1, 0))
    # False negatives
    fn = np.sum(np.where((pred != true) & (pred == 0), 1, 0))    

    # Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Precision
    precision = tp / (tp + fp)
    # Recall
    recall = tp / (tp + fn)
    # F1 score
    f1 = 2 * precision * recall / (precision + recall)

    return {'acc': acc, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1}

# # Plot prediction results for the knn model
# def plot_knn_results(ks, y, best_k, evaluation, title = 'Model', save_to = 'knn.png'):
#     plt.style.use('ggplot')
#     plt.figure(figsize=(10, 7))
#     plt.plot(ks, y, label = "Val Acc", c = "orange")
#     plt.scatter(best_k, y[best_k - 1], label=f"Best k = {best_k} (val acc: {y[best_k - 1]:.3f})", marker="*", s=100)
#     plt.xlabel('k (#)')
#     plt.ylabel('Validation Accuracy')
#     plt.legend()
#     plt.title(f'{title}\n(Eval Acc: {evaluation["acc"]:.3f}, Precision: {evaluation["precision"]:.3f}, Recall: {evaluation["recall"]:.3f}, f1: {evaluation["f1"]:.3f})')
#     plt.xticks(list(range(1, ks[-1] + 1)))
#     plt.savefig(save_to)
#     plt.show()

# Plot prediction
def plot_prediction(test_df, model, evaluation, title = 'Heart Disease Prediction w/ model', save_to = 'model.png', col0 = 'MaxHR', col1 = 'Age'):
    plt.figure(figsize=(7, 5))

    cols = [col0, col1]
    col_idx = [test_df.columns.get_loc(cols[0]), test_df.columns.get_loc(cols[1])]
    
    """ Draw prediction
    
        Draw prediction using mesh grid
    """
    x0_lin = np.linspace(test_df[cols[0]].min() - 1, test_df[cols[0]].max() + 1, 600)
    x1_lin = np.linspace(test_df[cols[1]].min() - 1, test_df[cols[1]].max() + 1, 600)
    x0_grid, x1_grid = np.meshgrid(x0_lin, x1_lin)
    x_points = np.c_[x0_grid.ravel(), x1_grid.ravel()]
    m = test_df.describe().loc['50%',:]
    x_mean = np.broadcast_to(m.to_numpy()[:-1], (len(x_points), len(m.to_numpy()[:-1]))).copy()
    x_mean[:, col_idx[0]] = x_points[:, 0]
    x_mean[:, col_idx[1]] = x_points[:, 1]
    y_grid = model.predict(x_mean).reshape(x0_grid.shape)

    # Define new colormap
    N = 100
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(151/256, 221/256, N)
    vals[:, 1] = np.linspace(191/256, 74/256, N)
    vals[:, 2] = np.linspace(180/256, 72/256, N)
    custom_cmp = ListedColormap(vals)
    plt.contourf(x0_grid, x1_grid, y_grid, alpha = 0.3, cmap = custom_cmp)

    """ Draw true data
    
        Draw true data
    """
    hd = test_df[test_df['HeartDisease'] == 1]
    nhd = test_df[test_df['HeartDisease'] == 0]
    plt.scatter(hd.loc[:, cols[0]], hd.loc[:, cols[1]], c='#DD4A48', label='Heart Disease', s = 50, alpha=0.6)
    plt.scatter(nhd.loc[:, cols[0]], nhd.loc[:, cols[1]], c='#97BFB4', label='No Heart Disease', s = 50, alpha=0.6)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.legend()
    plt.title(f'{title}\n(Eval Acc: {evaluation["acc"]:.3f}, Precision: {evaluation["precision"]:.3f}, Recall: {evaluation["recall"]:.3f}, f1: {evaluation["f1"]:.3f})')
    plt.savefig(save_to)
    plt.show()

# Plot training history
def plot_history(train_history, val_history, save_to = 'mlp_training.png'):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    for th, vh in zip(train_history, val_history):
        train_acc.append(th['acc'])
        train_loss.append(th['loss'])
        val_acc.append(vh['acc'])
        val_loss.append(vh['loss'])

    e = list(range(len(train_history)))
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    plt.subplot(1, 2, 1)
    plt.plot(e, train_loss, label="train")
    plt.plot(e, val_loss, label="val")
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(e, train_acc, label="train")
    plt.plot(e, val_acc, label="val")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig(save_to)
    plt.show()