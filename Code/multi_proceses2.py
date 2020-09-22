import os
from multiprocessing import Pool

def run_process(processess):
        os.system('python {}'.format(processess))

if __name__ == '__main__':
    processess = ('split1.py', 'split2.py', 'split3.py',  'split4.py', 'split5.py', 'split6.py', 'split7.py', 'split8.py', 'split9.py', 'split10.py',
            'split11.py', 'split12.py', 'split13.py',  'split14.py', 'split15.py', 'split16.py', 'split17.py', 'split18.py', 'split19.py', 'split20.py' )


    pool = Pool(processes = 25)
    pool.map(run_process, processess)