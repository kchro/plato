import subprocess

if __name__ == '__main__':
    fols = []
    with open('fols.txt', 'r') as f:
        for line in f:
            fols.append(line.rstrip())
    for fol in fols:
        with open('dataset_targets.txt', 'a') as f:
            f.write(fol+'\n')
        subprocess.call(['python runfol.py -f "%s" -r rules.all' % fol], shell=True)
        raise
