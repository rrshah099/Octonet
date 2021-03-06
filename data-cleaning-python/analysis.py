import sys
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.decomposition import PCA

from merge import load_and_augment_data


def get_X_y(df):
    """
    Given a dataframe with the 'Audio Clip' column loaded and all unneeded modes filtered, return two values:
    * the 2D ndarray of audios, with shape (num_audios, audio_length)
    * the 1D ndarray of binary labels
    """
    df_nonnull = df[~df['Audio Clip'].isnull()] # Some timepoints don't line up with the 20-second clips, these have null values
    labels = df_nonnull['Mode'].tolist()
    labels_binary = [(1 if label == 3 else 0) for label in labels]
    audios_list = [np.array(a.get_array_of_samples()) for a in df_nonnull['Audio Clip']]
    audios_array = np.stack(audios_list)
    return audios_array, labels_binary


if __name__ == '__main__':
    sp1 = '10112_R2_With time.xlsx'
    sp2 = '10812_R3_With time.xlsx'
    spreadsheet_dict = load_and_augment_data(sys.argv[1], files_to_load=[sp1, sp2])


    df1 = spreadsheet_dict[sp1]
    df2 = spreadsheet_dict[sp2]

    audios_array_1, labels_binary_1 = get_X_y(df1)
    audios_array_2, labels_binary_2 = get_X_y(df2)
    print('Binary labels (1 means mode 3, 0 means mode 15/18):', labels_binary_1)
    print('audios_array_1.shape:', audios_array_1.shape)

    pcamodel = PCA(n_components=50)
    lrmodel = LR()
    print('Fitting models using', sp1, 'as training data')
    reduced_1 = pcamodel.fit_transform(audios_array_1)
    print('Shape after PCA:', reduced_1.shape)
    lrmodel.fit(reduced_1, labels_binary_1)
    print('Logistic regression score on training data:')
    s = lrmodel.score(reduced_1, labels_binary_1)
    print(s)

    print('Computing test accuracy on', sp2)
    reduced_2 = pcamodel.transform(audios_array_2)
    s_test = lrmodel.score(reduced_2, labels_binary_2)
    print('Test accuracy:')
    print(s_test)

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    
    from sklearn.model_selection import train_test_split
    X = np.concatenate((audios_array_1, audios_array_2))
    y = np.concatenate((labels_binary_1, labels_binary_2))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    from sklearn.model_selection import train_test_split
    # X_train= audios_array_1 # Features
    # y_train= labels_binary_1  # Labels
    # X_test = audios_array_2
    # y_test = labels_binary_2
    from sklearn.ensemble import RandomForestClassifier
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    # from sklearn import svm
    # SVM = svm.LinearSVC()
    # SVM.fit(X_train,y_train)
    # y_pred=SVM.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
