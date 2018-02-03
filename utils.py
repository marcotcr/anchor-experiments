"""bla"""
# from __future__ import print_function
import copy
import collections
import sklearn
import numpy as np
# import string
import os
import sys
import lime
import lime.lime_tabular

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s#str(s)

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.iteritems():
        ret[ret == src] = target
    return ret

def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./datasets'):
    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'diabetes':
        categorical_features = [2, 3, 4, 5, 6, 7, 8, 10, 11, 18, 19, 20, 22,
                                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                47, 48]
        label_map = {'<30': 'YES', '>30': 'YES'}
        transformations = {
            49: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes/diabetic_data.csv'), -1, ',',
            features_to_use=range(2, 49),
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
    elif dataset_name == 'default':
        categorical_features = [2, 3, 4, 6, 7, 8, 9, 10, 11]
        dataset = load_csv_dataset(
                os.path.join(dataset_folder, 'default/default.csv'), -1, ',',
                features_to_use=range(1, 24),
                categorical_features=categorical_features, discretize=discretize,
                balance=balance)
    elif dataset_name == 'recidivism':
        features_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14]
        feature_names = ['Race', 'Alcohol', 'Junky', 'Supervised Release',
                         'Married', 'Felony', 'WorkRelease',
                         'Crime against Property', 'Crime against Person',
                         'Gender', 'Priors', 'YearsSchool', 'PrisonViolations',
                         'Age', 'MonthsServed', '', 'Recidivism']
        def violations_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'NO', '1': '1 to 5', '2': 'More than 5'})
        def priors_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [-1, 0, 5, float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'UNKNOWN', '1': 'NO', '2': '1 to 5', '3': 'More than 5'})
        transformations = {
            0: lambda x: replace_binary_values(x, ['Black', 'White']),
            1: lambda x: replace_binary_values(x, ['No', 'Yes']),
            2: lambda x: replace_binary_values(x, ['No', 'Yes']),
            3: lambda x: replace_binary_values(x, ['No', 'Yes']),
            4: lambda x: replace_binary_values(x, ['No', 'Married']),
            5: lambda x: replace_binary_values(x, ['No', 'Yes']),
            6: lambda x: replace_binary_values(x, ['No', 'Yes']),
            7: lambda x: replace_binary_values(x, ['No', 'Yes']),
            8: lambda x: replace_binary_values(x, ['No', 'Yes']),
            9: lambda x: replace_binary_values(x, ['Female', 'Male']),
            10: lambda x: priors_fn(x),
            12: lambda x: violations_fn(x),
            13: lambda x: (x.astype(float) / 12).astype(int),
            16: lambda x: replace_binary_values(x, ['No more crimes',
                                                    'Re-arrested'])
        }

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'recidivism/Data_1980.csv'), 16,
            feature_names=feature_names, discretize=discretize,
            features_to_use=features_to_use, balance=balance,
            feature_transformations=transformations, skip_first=True)
    elif dataset_name == 'lending':
        def filter_fn(data):
            to_remove = ['Does not meet the credit policy. Status:Charged Off',
               'Does not meet the credit policy. Status:Fully Paid',
               'In Grace Period', '-999', 'Current']
            for x in to_remove:
                data = data[data[:, 16] != x]
            return data
        bad_statuses = set(["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"])
        transformations = {
            16:  lambda x: np.array([y in bad_statuses for y in x]).astype(int),
            19:  lambda x: np.array([len(y) for y in x]).astype(int),
            6:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
            35:  lambda x: np.array([y.strip('%') if y else -1 for y in x]).astype(float),
        }
        features_to_use = [2, 12, 13, 19, 29, 35, 51, 52, 109]
        categorical_features = [12, 109]
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'lendingclub/LoanStats3a_securev1.csv'),
            16, ',',  features_to_use=features_to_use,
            feature_transformations=transformations, fill_na='-999',
            categorical_features=categorical_features, discretize=discretize,
            filter_fn=filter_fn, balance=True)
        dataset.class_names = ['Good Loan', 'Bad Loan']
    return dataset



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    ret.data = data
    return ret

def compute_exps(explain_fn, data):
    """explain_fn just takes in a piece of data
    data is a 2d array
    returns list of explanations
    """
    out = []
    for i, d in enumerate(data):
        if i % 100 == 0:
            print(i)
        out.append(explain_fn(d))
    return out


def get_reduced_explain_fn(explain_fn, predict_fn, **kwargs):
    def explain(data):
        return explain_fn(data, predict_fn, **kwargs)
    return explain

def greedy_pick_anchor(explanations, data, k=5, threshold=1):
    covered = {}
    n = float(data.shape[0])
    for i, (exp, d) in enumerate(zip(explanations, data)):
        fs = []
        for f, precision in zip(exp['feature'], exp['precision']):
            fs.append(f)
            if precision >= threshold:
                break
        fs = np.array(fs)
        if fs.shape[0] == 0:
            fs = np.array([exp['feature'][0]])
        covered[i] = set(
            np.all(data[:, fs] == d[fs], axis=1).nonzero()[0])
    chosen = []
    all_covered = set()
    for i in range(k):
        best = (-1, -1)
        for j in covered:
            gain = len(all_covered.union(covered[j]))
            if gain > best[1]:
                best = (j, gain)
        all_covered = all_covered.union(covered[best[0]])
        print(i, best[1] / n)
        chosen.append(best[0])
    return chosen


def evaluate_anchor(explanations, explanations_data, explanation_preds,
                    dataset, predictions, threshold=1):
    covered = {}
    n = float(dataset.shape[0])
    for i, (exp, d) in enumerate(zip(explanations, explanations_data)):
        fs = []
        for f, p in zip(exp['feature'], exp['precision']):
            fs.append(f)
            if p >= threshold:
                break
        fs = np.array(fs)
        covered[i] = set(
            np.all(dataset[:, fs] == d[fs], axis=1).nonzero()[0])
    # exp_range = range(len(explanations))
    n = float(dataset.shape[0])
    return precision_support_from_covered(n, covered, explanation_preds,
                                          predictions)


def precision_support_from_covered(n, covered, explanation_preds, predictions):
    predicted = 0.0
    predicted_right = 0.0
    for i in range(int(n)):
        votes = []
        for j in covered:
            if i in covered[j]:
                votes.append(explanation_preds[j])
        if votes:
            predicted += 1
            if np.random.choice(votes) == predictions[i]:
                predicted_right += 1
    if predicted == 0:
        return 1, 0
    precision = predicted_right / predicted
    support = predicted / n
    return precision, support


def greedy(submodular_fn, k, chosen=[]):
    chosen = copy.deepcopy(chosen)
    all_items = range(submodular_fn.num_items)
    current_value = 0
    z = 0
    while len(chosen) != k:
        best_gain = 0
        best_item = all_items[0]
        for i in all_items:
            gain = submodular_fn(chosen + [i]) - current_value
            if gain > best_gain:
                best_gain = gain
                best_item = i
        chosen.append(best_item)
        all_items.remove(best_item)
        current_value += best_gain
        print(z, current_value)
        z += 1
    return chosen


def submodular_fn(explanations, data, predictions, feature_value):
    """TODO: Detail this"""
    normalizer = sum(feature_value.values())

    def fnz(x):
        all_words = set()
        for doc in x:
            covered = []
            for f, v in explanations[doc].as_map()[predictions[doc]]:
                covered.append((f, data[doc][f]))
            all_words = all_words.union(covered)
        return sum([feature_value[w] for w in all_words]) / normalizer
    fnz.num_items = len(explanations)
    return fnz


def submodular_pick(explanations, data, predictions, k=5):
    # def submodular_temp():
    feature_value = collections.defaultdict(float)
    for exp, d, pred in zip(explanations, data, predictions):
        for f, v in exp.as_map()[pred]:
            feature_value[(f, d[f])] += np.abs(v)
    for f in feature_value:
        feature_value[f] = np.sqrt(feature_value[f])
    submodular = submodular_fn(explanations, data, predictions, feature_value)
    return greedy(submodular, k)


def submodular_coverage_pick(weights, vals, threshold, pred_threshold, binary, k,
                             verbose=False):
    val_covered = vals > pred_threshold
    if binary:
        val_covered += (1 - vals) > pred_threshold
    covered_a = val_covered * (weights >= threshold)
    covered = {}
    for i in range(covered_a.shape[1]):
        covered[i] = set(covered_a[:, i].nonzero()[0])
    n = float(val_covered.shape[0])
    chosen = []
    all_covered = set()
    for i in range(k):
        best = (-1, -1)
        chosen_set = set(chosen)
        for j in covered:
            if j in chosen_set:
                continue
            gain = len(all_covered.union(covered[j]))
            if gain > best[1]:
                best = (j, gain)
        all_covered = all_covered.union(covered[best[0]])
        if verbose:
            print(i, best[1] / n)
        chosen.append(best[0])
    return chosen

def compute_lime_weight_vals(explanations, exp_data, data):
    exp_range = range(len(explanations))
    distances = sklearn.metrics.pairwise_distances(data, exp_data,
                                                   metric='euclidean')
    kernel_width = np.sqrt(data.shape[1]) * .75

    def kernel(d):
        return np.sqrt(np.exp(-(d**2) / kernel_width ** 2))
    weights = kernel(distances)
    vals = np.zeros((len(data), len(explanations)))
    for i, d in enumerate(data):
        for j in exp_range:
            exp = explanations[j]['as_map']
            intercept = explanations[j]['intercept']
            val = intercept
            # val = 0.5
            for f, v in exp.iteritems():
                if d[f] == exp_data[j, f]:
                    val += v
            vals[i, j] = val
    return weights, vals

def evaluate_lime(weights, vals, explanation_preds, predictions, threshold, pred_threshold, binary):
    val_covered = vals > pred_threshold
    if binary:
        val_covered += (1 - vals) > pred_threshold
    covered = val_covered * (weights >= threshold)
    predicted = 0.
    predicted_right = 0.
    for d, val, pred in zip(covered, vals, predictions):
        cov = d.nonzero()[0]
        if cov.shape[0] == 0:
            continue
        chosen = np.random.choice(cov)
        val = val[chosen]
        predicted += 1
        pred_exp = explanation_preds[chosen]
        if binary:
            pred_exp = 1
        if binary and (1 - val) > pred_threshold:
            pred_exp = 1 - pred_exp
        if pred_exp == pred:
            predicted_right += 1
    n = float(weights.shape[0])
    if predicted == 0:
        return 1, 0
    # precision, support
    return predicted_right / predicted, predicted / n
