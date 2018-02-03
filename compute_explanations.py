from __future__ import print_function
import argparse
import pickle
import xgboost
import sklearn
import sklearn.neural_network
import utils
import anchor_tabular


def main():
    parser = argparse.ArgumentParser(description='Compute some explanations.')
    parser.add_argument('-d', dest='dataset', required=True,
                        choices=['adult', 'recidivism', 'lending'],
                        help='dataset to use')
    parser.add_argument('-e', dest='explainer', required=True,
                        choices=['lime', 'anchor'],
                        help='explainer, either anchor or lime')
    parser.add_argument('-m', dest='model', required=True,
                        choices=['xgboost', 'logistic', 'nn'],
                        help='model: xgboost, logistic or nn')
    parser.add_argument('-c', dest='checkpoint', required=False,
                        default=200, type=int,
                        help='checkpoint after this many explanations')
    parser.add_argument('-o', dest='output', required=True)

    args = parser.parse_args()
    dataset = utils.load_dataset(args.dataset, balance=True)
    ret = {}
    ret['dataset'] = args.dataset
    for x in ['train_idx', 'test_idx', 'validation_idx']:
            ret[x] = getattr(dataset, x)

    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train,
                  dataset.validation, dataset.labels_validation)

    if args.model == 'xgboost':
        c = xgboost.XGBClassifier(n_estimators=400, nthread=10, seed=1)
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    if args.model == 'logistic':
        c = sklearn.linear_model.LogisticRegression()
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
    if args.model == 'nn':
        c = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50))
        c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)


    ret['encoder'] = explainer.encoder
    ret['model'] = c
    ret['model_name'] = args.model

    def predict_fn(x):
        return c.predict(explainer.encoder.transform(x))

    def predict_proba_fn(x):
        return c.predict_proba(explainer.encoder.transform(x))

    print('Train', sklearn.metrics.accuracy_score(dataset.labels_train,
                                                  predict_fn(dataset.train)))
    print('Test', sklearn.metrics.accuracy_score(dataset.labels_test,
                                                 predict_fn(dataset.test)))
    threshold = 0.95
    tau = 0.1
    delta = 0.05
    epsilon_stop = 0.05
    batch_size = 100
    if args.explainer == 'anchor':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lucb_beam, c.predict, threshold=threshold,
            delta=delta, tau=tau, batch_size=batch_size / 2,
            sample_whole_instances=True,
            beam_size=10, epsilon_stop=epsilon_stop)
    elif args.explainer == 'lime':
        explain_fn = utils.get_reduced_explain_fn(
            explainer.explain_lime, c.predict_proba, num_features=5,
            use_same_dist=True)

    ret['exps'] = []
    for i, d in enumerate(dataset.validation, start=1):
        # print(i)
        if i % 100 == 0:
            print(i)
        if i % args.checkpoint == 0:
            print('Checkpointing')
            pickle.dump(ret, open(args.output + '.checkpoint', 'w'))
        ret['exps'].append(explain_fn(d))

    pickle.dump(ret, open(args.output, 'w'))


if __name__ == '__main__':
    main()
