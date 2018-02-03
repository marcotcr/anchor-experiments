from __future__ import print_function
import argparse
import os
import pickle
import sklearn
import numpy as np
import utils


def submodular_anchor_precrecall(z_anchor, dataset, preds_validation,
                                 preds_test, k):
    # returns picked, precisions, recalls
    picked = utils.greedy_pick_anchor(
        z_anchor['exps'],
        dataset.data[z_anchor['validation_idx']],
        k=k,
        threshold=1.1)
    precs = []
    recs = []
    for i in range(1, k + 1):
        exs = picked[:i]
        anchors = [z_anchor['exps'][i] for i in exs]
        data_anchors = dataset.data[z_anchor['validation_idx']][exs]
        pred_anchors = preds_validation[exs]
        prec, rec = utils.evaluate_anchor(
            anchors, data_anchors, pred_anchors,
            dataset.data[z_anchor['test_idx']], preds_test,
            threshold=1.1)
        precs.append(prec)
        recs.append(rec)
    return picked, precs, recs


def random_anchor_precrecall(z_anchor, dataset, preds_validation, preds_test,
                             k, do_all=False):
    # returns (anchor_prec, anchor_rec, anchor_prec_std, anchor_rec_std)
    temp_prec_anchor = []
    temp_rec_anchor = []
    all_ids = range(len(z_anchor['exps']))
    if do_all:
        all_exs = np.array(all_ids).reshape((-1, 1))
    else:
        all_exs = [np.random.choice(all_ids, k, replace=False) for x in range(500)]
    for exs in all_exs:
        anchors = [z_anchor['exps'][i] for i in exs]
        data_anchors = dataset.data[z_anchor['validation_idx']][exs]
        pred_anchors = preds_validation[exs]
        prec, rec = utils.evaluate_anchor(
            anchors, data_anchors, pred_anchors,
            dataset.data[z_anchor['test_idx']], preds_test,
            threshold=1.1)
        n = preds_test.shape[0]
        temp_prec_anchor.append(prec)
        temp_rec_anchor.append(rec)
    # print (predicted_right / predicted, predicted / total, 0, 0)
    return (np.mean(temp_prec_anchor), np.mean(temp_rec_anchor),
            np.std(temp_prec_anchor), np.std(temp_rec_anchor))


def random_until_k(random_fn, k):
    # random_fn takes k and returns prec, rec, prec_std, rec_std
    # returns (precisions, recalls, precisions_stds, recalls_stds)
    precs = []
    recs = []
    prec_stds = []
    rec_stds = []
    for i in range(1, k + 1):
        prec, rec, prec_std, rec_std = random_fn(i)
        precs.append(prec)
        recs.append(rec)
        prec_stds.append(prec_std)
        rec_stds.append(rec_std)
    return precs, recs, prec_stds, rec_stds


def submodular_lime_precrecall(z_lime, dataset, preds_validation, preds_test,
                               k, val_weights, val_vals, desired_precision=0,
                               to_change='distance', verbose=False,
                               threshold=0, pred_threshold=0.5):
    # returns (picked, precisions, recalls, threshold, pred_threshold)
    binary = np.bincount(preds_test).shape[0] <= 2
    if desired_precision != 0:
        thresholds = (np.linspace(0, 1, 101) if to_change == 'distance' else
                      np.linspace(0.5, 1, 51))
        for t in thresholds:
            if to_change == 'distance':
                threshold = t
            elif to_change == 'pred':
                pred_threshold = t
            else:
                print('Error: to_change must be pred or distance, it is',
                      to_change)
                quit()
            picked = utils.submodular_coverage_pick(val_weights, val_vals,
                                                    threshold, pred_threshold,
                                                    binary, k, verbose=False)
            exps = [z_lime['exps'][i] for i in picked]
            data_exps = dataset.data[z_lime['validation_idx']][picked]
            preds_exps = preds_validation[picked]
            w, v = utils.compute_lime_weight_vals(
                exps, data_exps, dataset.data[z_lime['test_idx']])
            prec, rec = utils.evaluate_lime(
                w, v, preds_exps, preds_test, threshold, pred_threshold,
                binary=binary)
            if verbose:
                print(t, prec, rec)
            if prec >= desired_precision:
                break
    picked_ = utils.submodular_coverage_pick(val_weights, val_vals,
                                             threshold, pred_threshold,
                                             binary, k)
    precs = []
    recs = []
    for i in range(1, k + 1):
        picked = picked_[:i]
        exps = [z_lime['exps'][i] for i in picked]
        data_exps = dataset.data[z_lime['validation_idx']][picked]
        preds_exps = preds_validation[picked]
        weights, vals = utils.compute_lime_weight_vals(
            exps, data_exps, dataset.data[z_lime['test_idx']])
        prec, rec = utils.evaluate_lime(weights, vals, preds_exps, preds_test,
                                        threshold, pred_threshold,
                                        binary=binary)
        precs.append(prec)
        recs.append(rec)
    return picked, precs, recs, threshold, pred_threshold

    return (prec, rec, threshold, pred_threshold)


def random_lime_precrecall(
    z_lime, dataset, preds_validation, preds_test, k, desired_precision=0,
    to_change='distance', do_all=False, verbose=False,
    threshold=0, pred_threshold=0.5):
    binary = np.bincount(preds_test).shape[0] <= 2
    all_ids = range(len(z_lime['exps']))
    if do_all:
        all_exs = np.array(all_ids).reshape((-1, 1))
    else:
        all_exs = [np.random.choice(all_ids, k, replace=False) for x in range(500)]
    ws = []
    vs = []
    for picked in all_exs:
        exps = [z_lime['exps'][i] for i in picked]
        data_exps = dataset.data[z_lime['validation_idx']][picked]
        w, v = utils.compute_lime_weight_vals(
            exps, data_exps, dataset.data[z_lime['test_idx']])
        ws.append(w)
        vs.append(v)

    thresholds = (np.linspace(0, 1, 101) if to_change == 'distance' else
                  np.linspace(0.5, 1, 51))
    for t in thresholds:
        if desired_precision != 0:
            if to_change == 'distance':
                threshold = t
            elif to_change == 'pred':
                pred_threshold = t
            else:
                print('Error: to_change must be pred or distance, it is',
                      to_change)
                quit()
        temp_prec_lime = []
        temp_rec_lime = []

        for picked, w, v in zip(all_exs, ws, vs):
            preds_exps = preds_validation[picked]
            prec, rec = utils.evaluate_lime(
                w, v, preds_exps, preds_test, threshold, pred_threshold,
                binary=binary)
            temp_prec_lime.append(prec)
            temp_rec_lime.append(rec)
        if verbose:
            print(t, np.mean(temp_prec_lime), np.mean(temp_rec_lime))
        if np.mean(temp_prec_lime) >= desired_precision:
            return (np.mean(temp_prec_lime), np.mean(temp_rec_lime),
                    np.std(temp_prec_lime), np.std(temp_rec_lime),
                    threshold, pred_threshold)
    return (1, 0, 0, 0, 1, 1)

def main():
    parser = argparse.ArgumentParser(description='Graphs')
    parser.add_argument(
        '-p', dest='pickle_folder',
        default='./out_pickles')
    parser.add_argument('-d', dest='dataset', required=True,
                        choices=['adult', 'recidivism', 'lending'],
                        help='dataset to use')
    parser.add_argument('-m', dest='model', required=True,
                        choices=['xgboost', 'logistic', 'nn'],
                        help='model: xgboost, logistic or nn')
    parser.add_argument(
        '-o', dest='output_folder',
        default='./results')

    args = parser.parse_args()
    dataset = utils.load_dataset(args.dataset, balance=True)
    dataset_name = args.dataset
    algorithm = args.model
    z_anchor = pickle.load(
        open(os.path.join(args.pickle_folder, '%s-anchor-%s' % (
            dataset_name, algorithm))))
    z_lime = pickle.load(
        open(os.path.join(args.pickle_folder, '%s-lime-%s' % (
            dataset_name, algorithm))))
    preds_validation = z_anchor['model'].predict(
        z_anchor['encoder'].transform(
            dataset.data[z_anchor['validation_idx']]))
    preds_test = z_anchor['model'].predict(
        z_anchor['encoder'].transform(
            dataset.data[z_anchor['test_idx']]))
    ret = {}
    ret['accuracy'] = sklearn.metrics.accuracy_score(
        dataset.labels[z_anchor['test_idx']], preds_test)
    print('accuracy', ret['accuracy'])

    print('Lime weights')
    val_weights, val_vals = utils.compute_lime_weight_vals(
        z_lime['exps'], dataset.data[z_lime['validation_idx']],
        dataset.data[z_lime['validation_idx']])

    print('Submodular anchor')
    picked, precs, recs = submodular_anchor_precrecall(
        z_anchor, dataset, preds_validation, preds_test, 10)
    ret['anchor_submodular'] = (picked, precs, recs)
    anchor_prec = precs[-1]

    print('Submodular lime pred')
    picked, precs, recs, t1, t2 = submodular_lime_precrecall(
        z_lime, dataset, preds_validation, preds_test, 10, val_weights,
        val_vals, desired_precision=anchor_prec, to_change='pred',
        verbose=True)

    ret['lime_pred_submodular'] = (picked, precs, recs)
    ret['lime_pred_submodular_threshold'] = t2

    print('Random anchor')
    (prec, cov, prec_std, cov_std) = random_anchor_precrecall(
        z_anchor, dataset, preds_validation, preds_test, 1, do_all=True)
    ret['anchor_1'] = (prec, cov, prec_std, cov_std)

    print('Random lime')
    (prec, cov, prec_std, cov_std, _, _) = random_lime_precrecall(
        z_lime, dataset, preds_validation, preds_test, k=1,
        desired_precision=0.0, to_change='distance', verbose=True,
        do_all=True)
    ret['lime_naive_1'] = (prec, cov, prec_std, cov_std)

    # print('Distance random lime')
    # (prec, cov, prec_std, cov_std, t1, t2) = random_lime_precrecall(
    #     z_lime, dataset, preds_validation, preds_test, k=1,
    #     desired_precision=0.0, to_change='distance', verbose=True,
    #     do_all=True, threshold=ret['lime_distance_submodular_threshold'])
    # ret['lime_distance_1'] = (prec, cov, prec_std, cov_std)
    # ret['lime_distance_1_threshold'] = t1

    print('Pred random lime')
    (prec, cov, prec_std, cov_std, t1, t2) = random_lime_precrecall(
        z_lime, dataset, preds_validation, preds_test, k=1,
        desired_precision=0.0, to_change='pred', verbose=True,
        do_all=True, pred_threshold=ret['lime_pred_submodular_threshold'])
    ret['lime_pred_1'] = (prec, cov, prec_std, cov_std)
    ret['lime_pred_1_threshold'] = t2

    def random_fn_lime(k):
        return random_lime_precrecall(
            z_lime, dataset, preds_validation, preds_test, k=k,
            desired_precision=0.0, to_change='pred', verbose=True,
            do_all=False,
            pred_threshold=ret['lime_pred_submodular_threshold'])[:4]

    def random_fn_anchor(k):
        return random_anchor_precrecall(
            z_anchor, dataset, preds_validation, preds_test, k, do_all=False)

    ret['anchor_random'] = random_until_k(random_fn_anchor, 10)
    ret['lime_pred_random'] = random_until_k(random_fn_lime, 10)

    path = os.path.join(args.output_folder, '%s-%s.pickle' % (
        dataset_name, algorithm))

    pickle.dump(ret, open(path, 'w'))

if __name__ == '__main__':
    main()
