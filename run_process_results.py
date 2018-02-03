import subprocess
datasets = ['adult', 'lending', 'recidivism']
models = ['xgboost', 'logistic', 'nn']
out = 'out_pickles'
out_folder = 'results'
for dataset in datasets:
    for model in models:
        outfile = '/tmp/%s-%s.log' % (dataset, model)
        print 'Outfile:', outfile
        outfile = open(outfile, 'w', 0)
        cmd = 'python process_results.py -d %s -p %s -m %s -o %s' % (
            dataset, out, model, out_folder)
        print cmd
        subprocess.Popen(cmd.split(), stdout=outfile)
