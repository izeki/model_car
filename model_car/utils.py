############################
# - compatibility with Python 3. This stuff from M. Brett's notebooks
# from __future__ import print_function  # print('me') instead of print 'me'
# The above seems to be slow to load, and is necessary to load in this file
# despite the import from kzpy if I want to use printing fully
#from __future__ import division  # 1/2 == 0.5, not 0
############################
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
######################




####################################
# exception format:
if False:
    import sys, traceback
    try:
        pass
    except Exception as e:
        print("********** Exception ***********************")
        traceback.print_exc(file=sys.stdout)
#        print(e.message, e.args)
#
####################################

def print_stars(n=1):
    for i in range(n):
        print("""*************************************************""")

# - import common modules
import os
import os.path
import shutil
import scipy
import scipy.io
import numpy as np  # the Python array package
import string
import glob
import time
import sys
import datetime
import random
try:
    import cPickle as pickle
except:
    import pickle
    print("utils.py: importing cPickle failed, using pickle instead.")
import re
import subprocess
from pprint import pprint
import serial
import math
import inspect
import fnmatch
try:
    import h5py
    from scipy.optimize import curve_fit
except:
    print("don't have h5py")
try:
    from termcolor import cprint
except:
    print("termcolor not installed")
    print("Try:  sudo pip install termcolor")
# - some definitions
import socket
host_name = socket.gethostname()
home_path = os.path.expanduser("~")
import getpass
username = getpass.getuser()
imread = scipy.misc.imread
imsave = scipy.misc.imsave
degrees = np.degrees
#opj = os.path.join
gg = glob.glob
arange = np.arange
os.environ['GLOG_minloglevel'] = '2'

def sgg(d):
    return sorted(gg(d),key=natural_keys)

def sggo(d,*args):
    a = opj(d,*args)
    return sgg(a)

shape = np.shape
randint = np.random.randint
#random = np.random.random # - this makes a conflict, so don't use it.
randn = np.random.randn
zeros = np.zeros
ones = np.ones
imresize = scipy.misc.imresize
reshape = np.reshape
mod = np.mod
array = np.array

abs = np.abs
def opj(*args):
    if len(args) == 0:
        args = ['']
    str_args = []
    for a in args:
        str_args.append(str(a))
    return os.path.join(*str_args)

def opjh(*args):
    return opj(home_path,opj(*args))

def opjD(*args):
    return opjh('Desktop',opj(*args))



def kzpy_utils_test():
    print('home_path = ' + home_path)
    print('Done.')


def copy_list_of_arrays(lst):
    nl = []
    for l in lst:
        nl.append(l.copy())
    return nl
    
def nps(x):
    return np.shape(x)


PRINT_COMMENTS = True
def CS_(comment,section=''):
    str = '# - '
    if len(section) > 0:
        str = str + section + ': '
    str = str + comment
    if PRINT_COMMENTS:
        print(str)


def zeroToOneRange(m):
    min_n = 1.0*np.min(m)
    return (1.0*m-min_n)/(1.0*np.max(m)-min_n)

z2o = zeroToOneRange

def dir_as_dic_and_list( path ):
    """Returns a dictionary and list of files and directories within the path.

    Keyword argument:
        path

    Certain types are ignored:
        .*      -- I want to avoid hidden files and directories.
        _*      -- I use underscore to indicate things to ignore.
        Icon*   -- The Icon? files are a nuisance created by
                  Google Drive that I also want to ignore."""
    return_dic = {}
    return_list = []
    for filename in os.listdir(path):
        if not filename[0] == '.': # ignore /., /.., and hidden directories and files
            if not filename[0] == '_': # ignore files starting with '_'
                if not filename[0:4] == 'Icon': # ignore Google Drive Icon things
                    return_dic[filename] = {}
                    return_list.append(filename)
    return_list.sort(key=natural_keys)
    return (return_dic,return_list)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def str_contains(st,str_list):
    for s in str_list:
        if not s in st:
            return False
    return True

def str_contains_one(st,str_list):
    for s in str_list:
        if s in st:
            return True
    return False

def select_keys(dic,str_list):
    key_list = []
    for k in dic.keys():
        if str_contains(k,str_list):
            key_list.append(k)
    key_list.sort(key=natural_keys) # if these are not sorted, the keys can be returned in different orders each time. If even and odd are compared, a different result can occur each time.
    return key_list


def unix(command_line_str, print_stdout=True, print_stderr=False,print_cmd=False):
    command_line_str = command_line_str.replace('~',home_path)
    p = subprocess.Popen(command_line_str.split(), stdout=subprocess.PIPE)
    stdout,stderr = p.communicate()
    if print_cmd:
        print(command_line_str)
    if print_stdout:
        print(stdout)
    if print_stderr:
        print(stderr)
#    return stdout,stderr
    return stdout.split('\n')

def d2s_spacer(args,spacer=' '):
    lst = []
    for e in args:
        lst.append(str(e))
    return spacer.join(lst)
def d2s(*args):
    '''
    e.g.,
    
    d2s('I','like',1,'or',[2,3,4])
    
    yields
    
    'I like 1 or [2, 3, 4]'
    
    d2c(1,2,3) => '1,2,3'
    d2f('/',1,2,3) => '1/2/3'
    '''
    return d2s_spacer(args)
def d2c(*args):
    return d2s_spacer(args,spacer=',')
def d2p(*args):
    return d2s_spacer(args,spacer='.')
def d2n(*args):
    return d2s_spacer(args,spacer='')
def d2f(*args):
    return d2s_spacer(args[1:],spacer=args[0])

def pd2s(*args):
    print(d2s(*args))


def dp(f,n=2):
    """
    get floats to the right number of decimal places, for display purposes
    """
    assert(n>=0)
    if n == 0:
        return int(np.round(f))
    f *= 10.0**n
    f = int(np.round(f))
    return f/(10.0**n)
   

def save_obj(obj, name ):
    if name.endswith('.pkl'):
        name = name[:-len('.pkl')]
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    if name.endswith('.pkl'):
        name = name[:-len('.pkl')]
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

lo = load_obj

def so(arg1,arg2):
    if type(arg1) == str and type(arg2) != str:
        save_obj(arg2,arg1)
        return
    if type(arg2) == str and type(arg1) != str:
        save_obj(arg1,arg2)
        return
    assert(False)



def psave(dic,data_path_key,path):
    save_obj(dic[data_path_key],opj(path,data_path_key))
    
def pload(dic,data_path_key,path):
    dic[data_path_key] = load_obj(opj(path,data_path_key))


def txt_file_to_list_of_strings(path_and_filename):
    f = open(path_and_filename,"r") #opens file with name of "test.txt"
    str_lst = []
    for line in f:
        str_lst.append(line.strip('\n'))
    return str_lst

def list_of_strings_to_txt_file(path_and_filename,str_lst,write_mode="w"):
    f = open(path_and_filename,write_mode)
    for s in str_lst:
        f.write(s+'\n')
    f.close()


def rebin(a, shape):
    '''
    from http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    '''
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def dict_to_sorted_list(d):
    l = []
    ks = sorted(d.keys(),key=natural_keys)
    for k in ks:
        l.append(d[k])
    return l


def get_sorted_keys_and_data(dict):
    skeys = sorted(dict.keys())
    sdata = []
    for k in skeys:
        sdata.append(dict[k])
    return skeys,sdata


def zscore(m,thresh=np.nan):
    z = m - np.mean(m)
    z /= np.std(m)
    if not np.isnan(thresh):
        z[z < -thresh] = -thresh
        z[z > thresh] = thresh
    return z

"""

%a - abbreviated weekday name
%A - full weekday name
%b - abbreviated month name
%B - full month name
%c - preferred date and time representation
%C - century number (the year divided by 100, range 00 to 99)
%d - day of the month (01 to 31)
%D - same as %m/%d/%y
%e - day of the month (1 to 31)
%g - like %G, but without the century
%G - 4-digit year corresponding to the ISO week number (see %V).
%h - same as %b
%H - hour, using a 24-hour clock (00 to 23)
%I - hour, using a 12-hour clock (01 to 12)
%j - day of the year (001 to 366)
%m - month (01 to 12)
%M - minute
%n - newline character
%p - either am or pm according to the given time value
%r - time in a.m. and p.m. notation
%R - time in 24 hour notation
%S - second
%t - tab character
%T - current time, equal to %H:%M:%S
%u - weekday as a number (1 to 7), Monday=1. Warning: In Sun Solaris Sunday=1
%U - week number of the current year, starting with the first Sunday as the first day of the first week
%V - The ISO 8601 week number of the current year (01 to 53), where week 1 is the first week that has at least 4 days in the current year, and with Monday as the first day of the week
%W - week number of the current year, starting with the first Monday as the first day of the first week
%w - day of the week as a decimal, Sunday=0
%x - preferred date representation without the time
%X - preferred time representation without the date
%y - year without a century (range 00 to 99)
%Y - year including the century
%Z or %z - time zone or name or abbreviation
%% - a literal % character


"""

def time_str(mode='FileSafe'):
    now = datetime.datetime.now()
    if mode=='FileSafe':
       return now.strftime('%d%b%y_%Hh%Mm%Ss')
    if mode=='Pretty':
       return now.strftime('%A, %d %b %Y, %r')


def zrn(c,verify=False,show_only=False):
    f = opjh('kzpy3/scratch/2015/12/scratch_script.py')
    t = txt_file_to_list_of_strings(f)
    ctr = 0
    u = '\n'.join(t)
    v = u.split('############\n')
    print('###########\n')
    print(v[c])
    if not show_only:
        if verify:
            d = raw_input('########### Do this? ')
            if d == 'y':
                exec(v[c],globals())
        else:
            exec(v[c],globals())



def getClipboardData():
    p = subprocess.Popen(['pbpaste'], stdout=subprocess.PIPE)
    retcode = p.wait()
    data = p.stdout.read()
    return data
gcd = getClipboardData
def setClipboardData(data):
    """
    setClipboardData
    """
    p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    p.stdin.write(data)
    p.stdin.close()
    retcode = p.wait()
scd = setClipboardData

def say(t):
    unix('say --interactive=/green -r 200 '+t)

def osa(s):
    os.system("""/usr/bin/osascript -e '""" + s + """'""")




def stowe_Desktop(dst=False):
    if dst==False:
        dst = opjh('Desktop_'+time_str())
    print(dst)
    unix('mkdir -p ' + dst)
    _,l = dir_as_dic_and_list(opjD(''))
    for i in l:
        shutil.move(opjD(i),dst)

def restore_Desktop(src):
    _,l = dir_as_dic_and_list(opjD(''))
    if len(l) > 0:
        print('**** Cannot restore Desktop because Desktop is not empty.')
        return False
    _,l = dir_as_dic_and_list(src)
    for i in l:
        shutil.move(opjh(src,i),opjD(''))

def advance(lst,e):
    lst.pop(0)
    lst.append(e)


def kill_ps(process_name_to_kill):
    ax_ps_lst = unix('ps ax',False)
    ps_lst = []
    for p in ax_ps_lst:
        if process_name_to_kill in p:
            ps_lst.append(p)
    pid_lst = []
    for i in range(len(ps_lst)):
        pid = int(ps_lst[i].split(' ')[1])
        pid_lst.append(pid)
    #print pid_lst
    for p in pid_lst:
        unix(d2s('kill',p))



def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system

        http://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result



def memory():
    """
    Get node total memory and memory usage
    http://stackoverflow.com/questions/17718449/determine-free-ram-in-python
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


def most_recent_file_in_folder(path,str_elements=[]):
    files = gg(opj(path,'*'))
    if len(files) == 0:
        return None
    candidates = []
    for f in files:
        is_candidate = True
        for s in str_elements:
            if s not in f:
                is_candidate = False
                break
        if is_candidate:
            candidates.append(f)
    mtimes = {}
    if len(candidates) == 0:
        return None
    for c in candidates:
        mtimes[os.path.getmtime(c)] = c

    mt = sorted(mtimes.keys())[-1]
    c = mtimes[mt]
    return c

def a_key(dic):
    keys = dic.keys()
    k = np.random.randint(len(keys))
    return keys[k]

def an_element(dic):
    return dic[a_key(dic)]

def apply_function_to_directories(fun,path,not_str_lst=[],and_str_lst=[],or_str_lst=[]):
    dirs = gg(opj(path,'*'))
    for d in dirs:
        ignore = False
        for i in not_str_lst:
            if i in d:
                ignore = True
                continue
        for r in and_str_lst:
            if r not in d:
                ignore = True
                continue
        if len(or_str_lst) > 0:
            or_ignore = True
        else:
            or_ignore = False
        for o in or_str_lst:
            if o in d:
                or_ignore = False
                continue
        if not ignore and not or_ignore:
            fun(d)
        else:
            print('*** Ignoring '+d)

            



t__0 = 0
def t_start():
    global t__0
    t__0 = time.time()

def t_end():
    print(time.time()-t__0)


def fn(path):
    """
    get filename part of path
    """
    return path.split('/')[-1]

def na(lst):
    return np.array(lst)

def to_range(e,a,b):
    if e < a:
        return a
    if e > b:
        return b
    return e

def in_range(e,a,b):
    if e >= a:
        if e <= b:
            return True
    return False

def nvidia_smi_continuous(t=0.1):
    while True:                                     
        unix('nvidia-smi')
        time.sleep(t)



class Timer:
    def __init__(self, time_s=0):
        self.time_s = time_s
        self.start_time = time.time()
    def check(self):
        if time.time() - self.start_time > self.time_s:
            return True
        else:
            return False
    def time(self):
        return time.time() - self.start_time
    def reset(self):
        self.start_time = time.time()
    def trigger(self):
        self.start_time = 0

  


def fname(path):
    return path.split('/')[-1]


def pname(path):
    p = path.split('/')[:-1]
    pstr = ""
    for s in p:
        if len(s)>0:
            pstr += '/' + s
    return pstr

def sequential_means(data,nn):
    a = array(data)
    d = []
    x = []
    n = min(len(a),nn)
    for i in range(0,len(a),n):
        d.append(a[i:i+n].mean())
        x.append(i+n/2.)
    return x,d


def tab_list_print(l,n=1,color=None,on_color=None):
    for e in l:
        s = ''
        for j in range(n):
            s += '\t'
        cprint(s+e,color,on_color)



def start_at(t):
    while time.time() < t:
        time.sleep(0.1)
        print(t-time.time())

try:
    import numbers
    def is_number(n):
        return isinstance(n,numbers.Number)
except:
    print("Don't have numbers module")



def str_replace(input_str,replace_dic):
    for r in replace_dic:
        input_str = input_str.replace(r,replace_dic[r])
    return input_str



def srtky(d):
    return sorted(d.keys())


def get_key_sorted_elements_of_dic(d,specific=None):
    ks = sorted(d.keys())
    els = []
    for k in ks:
        if specific == None:
            els.append(d[k])
        else:
            els.append(d[k][specific])
    return ks,els


def mean_of_upper_range(data,min_proportion,max_proportion):
    return array(sorted(data))[int(len(data)*min_proportion):int(len(data)*max_proportion)].mean()


def mean_exclude_outliers(data,n,min_proportion,max_proportion):
    """
    e.g.,

    L=lo('/media/karlzipser/ExtraDrive4/bair_car_data_new_28April2017/meta/direct_rewrite_test_11May17_16h16m49s_Mr_Blue/left_image_bound_to_data.pkl' )
    k,d = get_key_sorted_elements_of_dic(L,'encoder')
    d2=mean_of_upper_range_apply_to_list(d,30,0.33,0.66)
    CA();plot(k,d);plot(k,d2)
    
    """
    n2 = int(n/2)
    rdata = []
    len_data = len(data)
    for i in range(len_data):
        if i < n2:
            rdata.append(mean_of_upper_range(data[i:i-n2+n],min_proportion,max_proportion))
        elif i < len_data + n2:
            rdata.append(mean_of_upper_range(data[i-n2:i-n2+n],min_proportion,max_proportion))
        else:
            rdata.append(mean_of_upper_range(data[i-n2:i],min_proportion,max_proportion))
    return rdata





def pythonpaths(paths):
    for p in paths:
        sys.path.append(opjh(p))


def find_files_recursively(src,pattern,place='',FILES_ONLY=False,DIRS_ONLY=False):
    """
    https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """
    files = []
    folders = {}
    ctr = 0
    timer = Timer(5)
    if src[-1] != '/':
        src = src + '/'
    print(d2s('src =',src,'pattern =',pattern))
    for root, dirnames, filenames in os.walk(src):
        assert(not(FILES_ONLY and DIRS_ONLY))
        if FILES_ONLY:
            use_list = filenames
        elif DIRS_ONLY:
            use_list = dirnames
        else:
            use_list = filenames+dirnames
        for filename in fnmatch.filter(use_list, pattern):
            file = opj(root,filename)
            folder = pname(file).replace(place,'')
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(filename)
            ctr += 1
            if timer.check():
                print(d2s(time_str('Pretty'),ctr,'matches'))
                timer.reset()
    data = {}
    data['place'] = place
    data['paths'] = folders
    data['parent_folders'] = [fname(f) for f in folders.keys()]
    return data



def find_index_of_closest(val,lst):
    d = []
    for i in range(len(lst)):
        d.append(abs(lst[i]-val))
    return d.index(min(d))










def zaccess(d,alst,truncate=True,dic_show_ends=4):
    print(zdic_to_str(d,alst,truncate,dic_show_ends))
    for a in alst:
        #print a,d
        if type(d) != dict:
            break
        d = d[sorted(d.keys())[a]]
    return d

def zds(d,dic_show_ends,*alst):
    alst = list(alst)
    assert(dic_show_ends>1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    print(zdic_to_str(d,alst,False,dic_show_ends))



def zdl(d,dic_show_ends,*alst):
    alst = list(alst)
    assert(dic_show_ends>1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    list_of_strings_to_txt_file(opjh('kzpy3','zdl.txt'),zdic_to_str(d,alst,False,dic_show_ends).split('\n'))

def zdl(d,dic_show_ends,*alst):
    """
    https://stackoverflow.com/questions/2749796/how-to-get-the-original-variable-name-of-variable-passed-to-a-function
    """
    import inspect
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find('(') + 1:-1].split(',')
    names = []
    for i in args:
        if i.find('=') != -1:
            names.append(i.split('=')[1].strip())
        else:
            names.append(i)

    alst = list(alst)
    assert(dic_show_ends>1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    dic_str = zdic_to_str(d,alst,False,dic_show_ends)
    ks = []
    for a in alst:
        if type(d) != dict:
            break
        k = sorted(d.keys())[a]
        d = d[k]
        ks.append(k)
    out_str = ">> "+names[0]
    for k in ks:
        out_str += "['"+k+"']"
    cprint(out_str,'yellow')
    list_of_strings_to_txt_file(opjh('kzpy3','zdl.txt'),[out_str,names[0]]+dic_str.split('\n'))




def zda(d,dic_show_ends,*alst):
    """
    https://stackoverflow.com/questions/2749796/how-to-get-the-original-variable-name-of-variable-passed-to-a-function
    """
    import inspect
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find('(') + 1:-1].split(',')
    names = []
    for i in args:
        if i.find('=') != -1:
            names.append(i.split('=')[1].strip())
        else:
            names.append(i)

    zds(d,dic_show_ends,*alst)
    ks = []
    for a in alst:
        if type(d) != dict:
            break
        k = sorted(d.keys())[a]
        d = d[k]
        ks.append(k)
    out_str = ">> "+names[0]
    for k in ks:
        out_str += "['"+k+"']"
    cprint(out_str,'yellow')
    return d





def zlst_truncate(lst,show_ends=2):
    if show_ends == 0:
        return []
    if len(lst) > 2*show_ends:
        out_lst = lst[:show_ends] + ['...'] + lst[-show_ends:]
    else:
        out_lst = lst
    return out_lst

def zlst_to_str(lst,truncate=True,decimal_places=2,show_ends=2,depth=0,range_lst=[-2]):
    original_len = -1
    if truncate:
        original_len = len(lst)
        lst = zlst_truncate(lst,show_ends=show_ends)
    lst_str = d2n('\t'*(depth),"[")
    for i in range(len(lst)):
        e = lst[i]
        if type(e) == str:
            lst_str += e
        elif type(e) == int:
            lst_str += str(e)
        elif is_number(e):
            lst_str += str(dp(e,decimal_places))
        elif type(e) == list:
            lst_str += zlst_to_str(e,truncate=truncate,decimal_places=decimal_places,show_ends=show_ends)
        elif type(e) == dict:
            lst_str += zdic_to_str(d,range_lst,depth=depth+1)# zlst_to_str(e,truncate=truncate,decimal_places=decimal_places,show_ends=show_ends)
        else:
            lst_str += '???'
        if i < len(lst)-1:
            lst_str += ' '
    lst_str += ']'
    if original_len > 0:
        lst_str += d2n(' (len=',original_len,')')
    return lst_str

def zdic_to_str(d,range_lst,depth=0,dic_show_ends=4,dic_truncate=True):

    dic_str_lst = []

    sorted_keys = sorted(d.keys())
    
    this_range = range_lst[0]
    
    if type(this_range) == int:
        if this_range < 0:
            neg_two = False
            if this_range == -2:
                neg_two = True
            if dic_truncate:
                this_range = [0,min(dic_show_ends,len(sorted_keys))]
            else:
                this_range = [0,len(sorted_keys)]
            if neg_two:
                range_lst = range_lst + [-2]
        else:
            this_range = [this_range,this_range+1]

    if this_range[0] > 0:
        dic_str_lst.append(d2n('\t'*depth,'0) ...'))

    for i in range(this_range[0],this_range[1]):
        if i >= len(sorted_keys):
            return
        key = sorted_keys[i]
        value = d[key]

        dic_str_lst.append(d2n('\t'*depth,i,') ',key,':'))

        if isinstance(value,dict):
            if len(range_lst) > 1:
                dic_str_lst.append( zdic_to_str(value,range_lst[1:],depth=depth+1,dic_show_ends=dic_show_ends,dic_truncate=dic_truncate) )
            else:
                dic_str_lst.append(d2n('\t'*(depth+1),'...'))
        else:
            if type(value) == list:
                dic_str_lst.append(zlst_to_str(value,depth=depth+1,range_lst=range_lst[1:]))
            elif type(value) == np.ndarray:
                dic_str_lst.append(zlst_to_str(list(value),depth=depth+1,range_lst=range_lst[1:]))
            elif type(value) == str:
                dic_str_lst.append(d2s('\t'*(depth+1),str(value)))
            else:
                dic_str_lst.append(d2s('\t'*(depth+1),str(value),type(value)))
    if this_range[1] < len(sorted_keys):
        dic_str_lst.append(d2n('\t'*depth,'..',len(d)-1,')'))
    dic_str = ""
    for d in dic_str_lst:
        dic_str += d + "\n"

    return dic_str



def assert_disk_locations(locations):
    if type(locations) == str:
        locations = [locations]
    for l in locations:
        print(d2s("Checking for",l))
        if len(gg(l)) < 1:
            print(d2s(l,"not available!"))
            raw_input('Hit ctr-C')
            assert(False)
        print(d2s(l,'is there.\n'))







