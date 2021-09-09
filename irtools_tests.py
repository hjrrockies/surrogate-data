import matlab.engine as me
import numpy as np
import os
from scipy.io import loadmat

eng = me.start_matlab()

def load_test(name,n,**kwargs):
    n = str(n)
    if len(kwargs) == 0:
        directory = 'tests/'+name+'_'+n+'/'
    else:
        options_str = '_'.join(['_'.join([key,str(val)]) for key, val in kwargs.items()])
        directory = 'tests/'+name+'_'+n+'_'+options_str+'/'
    if os.path.isdir(directory):
        print('Test found in:',directory)
        A = loadmat(directory+'A.mat')
        b = loadmat(directory+'b.mat')
        x = loadmat(directory+'x.mat')
    else:
        print('Test not found. Attempting MATLAB call.')
        callstr = '[A,b,x,ProbInfo] = '
        callstr += name+'('
        callstr += n
        if len(kwargs) != 0:
            eng.evalc('options = struct;',nargout=0)
            for key, val in kwargs.items():
                if type(val) is str:
                    eng.evalc('options.'+key+'=\''+val+'\';',nargout=0)
                else:
                    eng.evalc('options.'+key+'='+str(val)+';',nargout=0)
            callstr += ',options'
        callstr += ')'
        print('MATLAB call:',callstr)
        eng.evalc(callstr,nargout=0)
        print('Saving .mat files...')
        if name == 'PRblur':
            print("checking for psfMatrix objects...")
            if 'PSF' not in kwargs.keys():
                kwargs['PSF'] = 'gauss'
            if kwargs['PSF'] != 'rotation':
                print("expanding psfMatrix objects...")
                eng.evalc("[U,S,V] = svd(A)")
                eng.evalc("A = (V*(U*diag(S))')'")

        os.makedirs(directory)
        eng.evalc("save(\'"+directory+"A.mat"+"\','A')")
        eng.evalc("save(\'"+directory+"b.mat"+"\','b')")
        eng.evalc("save(\'"+directory+"x.mat"+"\','x')")
    print('Loading .mat files...')
    A = loadmat(directory+'A.mat')['A']
    b = loadmat(directory+'b.mat')['b']
    x = loadmat(directory+'x.mat')['x']
    return A,b,x
