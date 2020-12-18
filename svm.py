# import torch
import argparse
# import torchvision
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt
## Hyperparameters - need to move? 
n_epochs = 10
batch_size_train = 60000
batch_size_test = 10000
learning_rate = 0.001
momentum = 0.5
log_interval = 10

random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

def load_data():
  # Load data
  ############################ change ##################################
  train_data = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))]))
  train_loader = torch.utils.data.DataLoader(train_data,
                 batch_size=batch_size_train, shuffle=True)

  test_data = torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize(
                                   (0.1307,), (0.3081,))
                               ]))
  test_loader = torch.utils.data.DataLoader(test_data,
                batch_size=batch_size_test, shuffle=True)

  train_x = next(iter(train_loader))[0].numpy()
  train_y = next(iter(train_loader))[1].numpy()
  test_x = next(iter(test_loader))[0].numpy()
  test_y = next(iter(test_loader))[1].numpy()

  n, c, w, h = train_x.shape
  train_x = np.reshape(train_x, (n, w*h)) 
  # print(train_x.shape)

  n, c, w, h = test_x.shape
  test_x = np.reshape(test_x, (n, w*h)) 
  # print(test_x.shape)

  # print(train_x.shape)
  # print(train_y.shape)
  # print(test_x.shape)
  # print(test_y.shape)

  return train_x, train_y, test_x, test_y

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_red', type=str, default='none', help='{none, pca, lda}')
    args = parser.parse_args()
    red = args.dem_red

    # Load Data
    # train_data, train_label, test_data, test_label = load_data()
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    print(train_data.shape)
    print(test_data.shape)

    train_data = train_data.reshape((60000,784))
    test_data = test_data.reshape((10000,784))
    # train_label = train_label.reshape((60000,))
    # test_label = test_label.reshape((10000,))

    print(train_data.shape)
    print(test_data.shape)
    train_data = train_data/255.0
    test_data = test_data/255.0 
    # train_data = scale(train_data)
    # test_data = scale(test_data)
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    # Dimensionality Reduction
    if red == 'PCA':
      print("Applying PCA")
      n_components = 5 
      pca = PCA(n_components=n_components, svd_solver='randomized',
                whiten=True)
      train_data = pca.fit_transform(train_data)
      test_data = pca.transform(test_data)
      print("Explained Variance ration:", pca.explained_variance_ratio_.sum())

    if red == 'LDA':
      print("Applying LDA")
      lda = LDA(n_components=16)
      train_data = lda.fit_transform(train_data, train_label)
      test_data = lda.transform(test_data)

    # Linear Kernel
    print("Training Linear Classifier...")
    clf = SVC(kernel='linear', cache_size=2000)
    clf.fit(train_data[:1000, :], train_label[:1000])
    print("Training complete!")

    pred_labels = clf.predict(test_data)

    model_acc = clf.score(test_data, test_label)
    test_acc = accuracy_score(test_label, pred_labels)
    conf_mat = confusion_matrix(test_label,pred_labels)

    print("Model Performance:")
    print("Model accuracy: ", model_acc)
    print("Test accuracy: ", model_acc)
    print("Confusion matrix:")
    print(model_acc)

    fig = plt.figure(figsize=(10, 10)) # Set Figure
    # Plot Confusion matrix
    sns.heatmap(conf_mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Greens)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values');
    # plt.show();
    plt.savefig('linear.png')
    plt.close()


    # Polynomial Kernel
    print("Training Polynomial Classifier...")
    clf = SVC(kernel='poly')
    clf.fit(train_data, train_label)
    print("Training complete!")

    pred_labels = clf.predict(test_data)

    model_acc = clf.score(test_data, test_label)
    test_acc = accuracy_score(test_label, pred_labels)
    conf_mat = confusion_matrix(test_label,pred_labels)

    print("Model Performance:")
    print("Model accuracy: ", model_acc)
    print("Test accuracy: ", model_acc)
    print("Confusion matrix:")
    print(model_acc)

    fig = plt.figure(figsize=(10, 10)) # Set Figure
    # Plot Confusion matrix
    sns.heatmap(conf_mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Greens)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values');
    # plt.show();
    plt.savefig(f'poly.png')
    plt.close()

    # RBF Kernel
    print("Training RBF Classifier...")
    clf = SVC(kernel='rbf')
    clf.fit(train_data, train_label)
    print("Training complete!")

    pred_labels = clf.predict(test_data)

    model_acc = clf.score(test_data, test_label)
    test_acc = accuracy_score(test_label, pred_labels)
    conf_mat = confusion_matrix(test_label,pred_labels)

    print("Model Performance:")
    print("Model accuracy: ", model_acc)
    print("Test accuracy: ", model_acc)
    print("Confusion matrix:")
    print(model_acc)

    fig = plt.figure(figsize=(10, 10)) # Set Figure
    # Plot Confusion matrix
    sns.heatmap(conf_mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Greens)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values');
    # plt.show();
    plt.savefig(f'rbf.png')
    plt.close()