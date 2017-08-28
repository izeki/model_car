"""Utilities imported from kzpy3."""
############################
# - compatibility with Python 3. This stuff from M. Brett's notebooks
# from __future__ import print_function  # print('me') instead of print 'me'
# The above seems to be slow to load, and is necessary to load in this file
# despite the import from kzpy if I want to use printing fully
# from __future__ import division  # 1/2 == 0.5, not 0
############################
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
######################

import_list = [
    'os',
    'os.path',
    'shutil',
    'scipy',
    'scipy.io',
    'string',
    'glob',
    'time',
    'sys',
    'datetime',
    'random',
    'cPickle',
    're',
    'subprocess',
    'serial',
    'math',
    'inspect',
    'fnmatch',
    'h5py',
    'socket',
    'getpass',
    'numbers']
import_from_list = [['FROM', 'pprint', 'pprint'], [
    'FROM', 'scipy.optimize', 'curve_fit'], ['FROM', 'termcolor', 'cprint']]
import_as_list = [['AS', 'numpy', 'np'], ['AS', 'cPickle', 'pickle']]

for im in import_list + import_from_list + import_as_list:
    if isinstance(im, str):
        try:
            exec('import ' + im)
            #print("imported "+im)
        except BaseException:
            print('Failed to import ' + im)
    else:
        assert(type(im)) == list
        if im[0] == 'FROM':
            try:
                exec('from ' + im[1] + ' import ' + im[2])
                #print("from "+im[1]+" imported "+im[2])
            except BaseException:
                print('Failed to from ' + im[1] + ' import ' + im[2])
        else:
            assert(im[0] == 'AS')
            try:
                exec('import ' + im[1] + ' as ' + im[2])
                #print("imported "+im[1]+" as "+im[2])
            except BaseException:
                print('Failed to import ' + im[1] + ' as ' + im[2])

#print("*** Note, kzpy3/teg2/bashrc now does: 'export PYTHONSTARTUP=~/kzpy3/vis2.py' ***")


####################################
# exception format:
if False:
    try:
        pass
    except Exception as e:
        print("********** Exception ***********************")
        print(e.message, e.args)
#
####################################


def print_stars(n=1):
    for i in range(n):
        print("""*************************************************""")


def print_stars0(n=1):
    print_stars()
    print("*")


def print_stars1(n=1):
    print("*")
    print_stars()


host_name = socket.gethostname()
home_path = os.path.expanduser("~")
username = getpass.getuser()

imread = scipy.misc.imread
imsave = scipy.misc.imsave
degrees = np.degrees


arange = np.arange
os.environ['GLOG_minloglevel'] = '2'

gg = glob.glob


def sgg(d):
    return sorted(gg(d), key=natural_keys)


def sggo(d, *args):
    a = opj(d, *args)
    return sgg(a)


shape = np.shape
randint = np.random.randint
# random = np.random.random # - this makes a conflict, so don't use it.
randn = np.random.randn
zeros = np.zeros
ones = np.ones
imresize = scipy.misc.imresize
reshape = np.reshape
mod = np.mod
array = np.array


def opj(*args):
    if len(args) == 0:
        args = ['']
    str_args = []
    for a in args:
        str_args.append(str(a))
    return os.path.join(*str_args)


def opjh(*args):
    return opj(home_path, opj(*args))


def opjD(*args):
    return opjh('Desktop', opj(*args))


media_path = opj('/media', username)


def rlen(a):
    return range(len(a))


PRINT_COMMENTS = True


def CS_(comment, section=''):
    str = '# - '
    if len(section) > 0:
        str = str + section + ': '
    str = str + comment
    if PRINT_COMMENTS:
        print(str)


def zeroToOneRange(m):
    min_n = 1.0 * np.min(m)
    return (1.0 * m - min_n) / (1.0 * np.max(m) - min_n)


z2o = zeroToOneRange


def dir_as_dic_and_list(path):
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
        if not filename[0] == '.':  # ignore /., /.., and hidden directories and files
            if not filename[0] == '_':  # ignore files starting with '_'
                if not filename[0:4] == 'Icon':  # ignore Google Drive Icon things
                    return_dic[filename] = {}
                    return_list.append(filename)
    return_list.sort(key=natural_keys)
    return (return_dic, return_list)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def str_contains(st, str_list):
    for s in str_list:
        if s not in st:
            return False
    return True


def str_contains_one(st, str_list):
    for s in str_list:
        if s in st:
            return True
    return False


def unix(
        command_line_str,
        print_stdout=True,
        print_stderr=False,
        print_cmd=False):
    command_line_str = command_line_str.replace('~', home_path)
    p = subprocess.Popen(command_line_str.split(), stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if print_cmd:
        print(command_line_str)
    if print_stdout:
        print(stdout)
    if print_stderr:
        print(stderr)
#    return stdout,stderr
    return stdout.split('\n')


def d2s_spacer(args, spacer=' '):
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
    return d2s_spacer(args, spacer=',')


def d2p(*args):
    return d2s_spacer(args, spacer='.')


def d2n(*args):
    return d2s_spacer(args, spacer='')


def d2f(*args):
    return d2s_spacer(args[1:], spacer=args[0])


def pd2s(*args):
    print(d2s(*args))


def dp(f, n=2):
    """
    get floats to the right number of decimal places, for display purposes
    """
    assert(n >= 0)
    if n == 0:
        return int(np.round(f))
    f *= 10.0**n
    f = int(np.round(f))
    return f / (10.0**n)


def save_obj(obj, name):
    if name.endswith('.pkl'):
        name = name[:-len('.pkl')]
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if name.endswith('.pkl'):
        name = name[:-len('.pkl')]
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


lo = load_obj


def so(arg1, arg2):
    if isinstance(arg1, str) and not isinstance(arg2, str):
        save_obj(arg2, arg1)
        return
    if isinstance(arg2, str) and not isinstance(arg1, str):
        save_obj(arg1, arg2)
        return
    assert(False)


def psave(dic, data_path_key, path):
    save_obj(dic[data_path_key], opj(path, data_path_key))


def pload(dic, data_path_key, path):
    dic[data_path_key] = load_obj(opj(path, data_path_key))


def txt_file_to_list_of_strings(path_and_filename):
    f = open(path_and_filename, "r")  # opens file with name of "test.txt"
    str_lst = []
    for line in f:
        str_lst.append(line.strip('\n'))
    return str_lst


def list_of_strings_to_txt_file(path_and_filename, str_lst, write_mode="w"):
    f = open(path_and_filename, write_mode)
    for s in str_lst:
        f.write(s + '\n')
    f.close()


def rebin(a, shape):
    '''
    from http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    '''
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def dict_to_sorted_list(d):
    l = []
    ks = sorted(d.keys(), key=natural_keys)
    for k in ks:
        l.append(d[k])
    return l


def get_sorted_keys_and_data(dict):
    skeys = sorted(dict.keys())
    sdata = []
    for k in skeys:
        sdata.append(dict[k])
    return skeys, sdata


def zscore(m, thresh=np.nan):
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
    if mode == 'FileSafe':
        return now.strftime('%d%b%y_%Hh%Mm%Ss')
    if mode == 'Pretty':
        return now.strftime('%A, %d %b %Y, %r')


def zrn(c, verify=False, show_only=False):
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
                exec(v[c], globals())
        else:
            exec(v[c], globals())


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
    unix('say --interactive=/green -r 200 ' + t)


def stowe_Desktop(dst=False):
    if not dst:
        dst = opjh('Desktop_' + time_str())
    print(dst)
    unix('mkdir -p ' + dst)
    _, l = dir_as_dic_and_list(opjD(''))
    for i in l:
        shutil.move(opjD(i), dst)


def restore_Desktop(src):
    _, l = dir_as_dic_and_list(opjD(''))
    if len(l) > 0:
        print('**** Cannot restore Desktop because Desktop is not empty.')
        return False
    _, l = dir_as_dic_and_list(src)
    for i in l:
        shutil.move(opjh(src, i), opjD(''))


def advance(lst, e):
    lst.pop(0)
    lst.append(e)


def kill_ps(process_name_to_kill):
    ax_ps_lst = unix('ps ax', False)
    ps_lst = []
    for p in ax_ps_lst:
        if process_name_to_kill in p:
            ps_lst.append(p)
    pid_lst = []
    for i in range(len(ps_lst)):
        pid = int(ps_lst[i].split(' ')[1])
        pid_lst.append(pid)
    # print pid_lst
    for p in pid_lst:
        unix(d2s('kill', p))


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


def most_recent_file_in_folder(path, str_elements=[], ignore_str_elements=[]):
    files = gg(opj(path, '*'))
    if len(files) == 0:
        return None
    candidates = []
    for f in files:
        is_candidate = True
        for s in str_elements:
            if s not in f:
                is_candidate = False
                break
        for s in ignore_str_elements:
            if s in f:
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


def fn(path):
    """
    get filename part of path
    """
    return path.split('/')[-1]


def to_range(e, a, b):
    if e < a:
        return a
    if e > b:
        return b
    return e


def in_range(e, a, b):
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
        if len(s) > 0:
            pstr += '/' + s
    return pstr


def sequential_means(data, nn):
    a = array(data)
    d = []
    x = []
    n = min(len(a), nn)
    for i in range(0, len(a), n):
        d.append(a[i:i + n].mean())
        x.append(i + n / 2.)
    return x, d


def tab_list_print(l, n=1, color=None, on_color=None):
    for e in l:
        s = ''
        for j in range(n):
            s += '\t'
        cprint(s + e, color, on_color)


def start_at(t):
    while time.time() < t:
        time.sleep(0.1)
        print(t - time.time())


try:
    import numbers

    def is_number(n):
        return isinstance(n, numbers.Number)
except BaseException:
    print("Don't have numbers module")


def str_replace(input_str, replace_dic):
    for r in replace_dic:
        input_str = input_str.replace(r, replace_dic[r])
    return input_str


def srtky(d):
    return sorted(d.keys())


def get_key_sorted_elements_of_dic(d, specific=None):
    ks = sorted(d.keys())
    els = []
    for k in ks:
        if specific is None:
            els.append(d[k])
        else:
            els.append(d[k][specific])
    return ks, els


def mean_of_upper_range(data, min_proportion, max_proportion):
    return array(sorted(data))[int(len(data) *
                                   min_proportion):int(len(data) *
                                                       max_proportion)].mean()


def mean_exclude_outliers(data, n, min_proportion, max_proportion):
    """
    e.g.,

    L=lo('/media/karlzipser/ExtraDrive4/bair_car_data_new_28April2017/meta/direct_rewrite_test_11May17_16h16m49s_Mr_Blue/left_image_bound_to_data.pkl' )
    k,d = get_key_sorted_elements_of_dic(L,'encoder')
    d2=mean_of_upper_range_apply_to_list(d,30,0.33,0.66)
    CA();plot(k,d);plot(k,d2)

    """
    n2 = int(n / 2)
    rdata = []
    len_data = len(data)
    for i in range(len_data):
        if i < n2:
            rdata.append(mean_of_upper_range(
                data[i:i - n2 + n], min_proportion, max_proportion))
        elif i < len_data + n2:
            rdata.append(mean_of_upper_range(
                data[i - n2:i - n2 + n], min_proportion, max_proportion))
        else:
            rdata.append(mean_of_upper_range(
                data[i - n2:i], min_proportion, max_proportion))
    return rdata


def meo(data, n):
    return mean_exclude_outliers(data, n, 1 / 3.0, 2 / 3.0)


def pythonpaths(paths):
    for p in paths:
        sys.path.append(opjh(p))


def find_files_recursively(src, pattern, FILES_ONLY=False, DIRS_ONLY=False):
    """
    https://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """
    files = []
    folders = {}
    ctr = 0
    timer = Timer(5)
    if src[-1] != '/':
        src = src + '/'
    print(d2s('src =', src, 'pattern =', pattern))
    for root, dirnames, filenames in os.walk(src):
        assert(not(FILES_ONLY and DIRS_ONLY))
        if FILES_ONLY:
            use_list = filenames
        elif DIRS_ONLY:
            use_list = dirnames
        else:
            use_list = filenames + dirnames
        for filename in fnmatch.filter(use_list, pattern):
            file = opj(root, filename)
            folder = pname(file).replace(place, '')
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(filename)
            ctr += 1
            if timer.check():
                print(d2s(time_str('Pretty'), ctr, 'matches'))
                timer.reset()
    data = {}
    data['paths'] = folders
    data['parent_folders'] = [fname(f) for f in folders.keys()]
    return data


def find_index_of_closest(val, lst):
    d = []
    for i in range(len(lst)):
        d.append(abs(lst[i] - val))
    return d.index(min(d))


##################################
#


ZD_Dictionary = None
ZD_Dictionary_name = '<no name>'
ZD_dic_show_ends = 24


def zaccess(d, alst, truncate=True, dic_show_ends=4):
    print(zdic_to_str(d, alst, truncate, dic_show_ends))
    for a in alst:
        # print a,d
        if not isinstance(d, dict):
            break
        d = d[sorted(d.keys())[a]]
    return d


def zds(d, dic_show_ends, *alst):
    alst = list(alst)
    assert(dic_show_ends > 1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    print(zdic_to_str(d, alst, False, dic_show_ends))


def _zdl(d, dic_show_ends, *alst):
    alst = list(alst)
    assert(dic_show_ends > 1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    list_of_strings_to_txt_file(
        opjh(
            'kzpy3', 'zdl.txt'), zdic_to_str(
            d, alst, False, dic_show_ends).split('\n'))


def zdl(d, dic_show_ends, *alst):
    """
    https://stackoverflow.com/questions/2749796/how-to-get-the-original-variable-name-of-variable-passed-to-a-function
    """
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
    """
    alst = list(alst)
    assert(dic_show_ends > 1)
    if len(alst) == 0:
        print("zds(d,dic_show_ends,*alst), but len(alst) == 0")
    dic_str = zdic_to_str(d, alst, False, dic_show_ends)
    ks = []
    for a in alst:
        if not isinstance(d, dict):
            break
        k = sorted(d.keys())[a]
        d = d[k]
        ks.append(k)
    out_str = ">> " + ZD_Dictionary_name  # names[0]
    for k in ks:
        if is_number(k) or isinstance(k, tuple):
            out_str += '[' + str(k) + ']'
        else:
            out_str += "['" + k + "']"
    cprint(out_str, 'yellow')
    list_of_strings_to_txt_file(
        opjh(
            'kzpy3', 'zdl.txt'), [
            out_str, ZD_Dictionary_name] + dic_str.split('\n'))


def zdset(d, dic_show_ends=24):
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
    global ZD_Dictionary, ZD_Dictionary_name, ZD_dic_show_ends
    ZD_Dictionary = d
    ZD_Dictionary_name = names[0]
    ZD_dic_show_ends = dic_show_ends


def zd(*alst):
    alst = list(alst)
    if len(alst) == 0:
        alst = [-1]
    zdl(ZD_Dictionary, ZD_dic_show_ends, *alst)


def zda(d, dic_show_ends, *alst):
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

    zds(d, dic_show_ends, *alst)
    ks = []
    for a in alst:
        if not isinstance(d, dict):
            break
        k = sorted(d.keys())[a]
        d = d[k]
        ks.append(k)
    out_str = ">> " + names[0]
    for k in ks:
        if is_number(k):
            out_str += '[' + str(k) + ']'
        else:
            out_str += "['" + k + "']"
    cprint(out_str, 'yellow')
    return d


def zlst_truncate(lst, show_ends=2):
    if show_ends == 0:
        return []
    if len(lst) > 2 * show_ends:
        out_lst = lst[:show_ends] + ['...'] + lst[-show_ends:]
    else:
        out_lst = lst
    return out_lst


def zlst_to_str(lst,
                truncate=True,
                decimal_places=2,
                show_ends=2,
                depth=0,
                range_lst=[-2]):
    original_len = -1
    if truncate:
        original_len = len(lst)
        lst = zlst_truncate(lst, show_ends=show_ends)
    lst_str = d2n('\t' * (depth), "[")
    for i in range(len(lst)):
        e = lst[i]
        if isinstance(e, str):
            lst_str += e
        elif isinstance(e, int):
            lst_str += str(e)
        elif is_number(e):
            lst_str += str(dp(e, decimal_places))
        elif isinstance(e, list):
            lst_str += zlst_to_str(e,
                                   truncate=truncate,
                                   decimal_places=decimal_places,
                                   show_ends=show_ends)
        elif isinstance(e, dict):
            # zlst_to_str(e,truncate=truncate,decimal_places=decimal_places,show_ends=show_ends)
            lst_str += zdic_to_str(e, range_lst, depth=depth + 1)
        else:
            lst_str += '???'
        if i < len(lst) - 1:
            lst_str += ' '
    lst_str += ']'
    if original_len > 0:
        lst_str += d2n(' (len=', original_len, ')')
    return lst_str


def zdic_to_str(d, range_lst, depth=0, dic_show_ends=4, dic_truncate=True):

    dic_str_lst = []

    sorted_keys = sorted(d.keys())

    this_range = range_lst[0]

    if isinstance(this_range, int):
        if this_range < 0:
            neg_two = False
            if this_range == -2:
                neg_two = True
            if dic_truncate:
                this_range = [0, min(dic_show_ends, len(sorted_keys))]
            else:
                this_range = [0, len(sorted_keys)]
            if neg_two:
                range_lst = range_lst + [-2]
        else:
            this_range = [this_range, this_range + 1]

    if this_range[0] > 0:
        dic_str_lst.append(d2n('\t' * depth, '0) ...'))

    for i in range(this_range[0], this_range[1]):
        if i >= len(sorted_keys):
            return
        key = sorted_keys[i]
        value = d[key]

        dic_str_lst.append(d2n('\t' * depth, i, ') ', key, ':'))

        if isinstance(value, dict):
            if len(range_lst) > 1:
                dic_str_lst.append(zdic_to_str(value,
                                               range_lst[1:],
                                               depth=depth + 1,
                                               dic_show_ends=dic_show_ends,
                                               dic_truncate=dic_truncate))
            else:
                dic_str_lst.append(d2n('\t' * (depth + 1), '...'))
        else:
            if isinstance(value, list):
                dic_str_lst.append(zlst_to_str(
                    value, depth=depth + 1, range_lst=range_lst[1:]))
            elif isinstance(value, np.ndarray):
                dic_str_lst.append(zlst_to_str(
                    list(value), depth=depth + 1, range_lst=range_lst[1:]))
            elif isinstance(value, str):
                dic_str_lst.append(d2s('\t' * (depth + 1), str(value)))
            else:
                dic_str_lst.append(
                    d2s('\t' * (depth + 1), str(value), type(value)))
    if this_range[1] < len(sorted_keys):
        dic_str_lst.append(d2n('\t' * depth, '..', len(d) - 1, ')'))
    dic_str = ""
    for d in dic_str_lst:
        dic_str += d + "\n"

    return dic_str
#
#############################


def assert_disk_locations(locations):
    if isinstance(locations, str):
        locations = [locations]
    for l in locations:
        print(d2s("Checking for", l))
        if len(gg(l)) < 1:
            print(d2s(l, "not available!"))
            raw_input('Hit ctr-C')
            assert(False)
        print(d2s(l, 'is there.\n'))


def XX(in_str):
    eqn = in_str.split('=')
    var_name = eqn[0].replace(' ', '')
    elements = eqn[1]
    elements = in_str.split('/')
    exec_lst = []
    exec_lst.append(elements[0])
    for i in range(1, len(elements)):
        quote = "'"
        if '`' in elements[i]:
            quote = ""
        exec_lst.append(
            ('[' +
             quote +
             elements[i] +
             quote +
             ']').replace(
                '`',
                ''))
    exec_str = var_name + " = " + ("".join(exec_lst)).replace(' ', '')
    return exec_str


def remove_functions_from_dic(d):
    for k in d.keys():
        if callable(d[k]):
            d[k] = 'FUNCTION_PLACEHOLDER'


def even_len(d):
    l = d['l']
    return np.mod(len(l), 2) == 0


def args_to_dic(d):
    pargs = d['pargs']
    if isinstance(pargs, str):
        pargs = pargs.split(' ')
    assert(even_len({'l': pargs}))
    rargs = {}
    for i in range(0, len(pargs), 2):
        assert(pargs[i][0] == '-')
        k = pargs[i][1:]
        val = pargs[i + 1]
        exec(d2n("rargs['", k, "'] = ", "'", val, "'"))
        if isinstance(rargs[k],
                      str) and rargs[k][0] == '{' and rargs[k][-1] == '}':
            exec('rargs[k] = ' + rargs[k])
        elif isinstance(rargs[k], str) and rargs[k][0] == '[' and rargs[k][-1] == ']':
            exec('rargs[k] = ' + rargs[k])
    return rargs


def translate_args(d):
    translation_dic = d['translation_dic']
    argument_dictionary = d['argument_dictionary']
    for k in translation_dic.keys():
        v = translation_dic[k]
        translation_dic['-' + v] = v
    new_dictionary = {}
    for k in argument_dictionary.keys():
        if k in translation_dic.keys():
            new_dictionary[translation_dic[k]] = argument_dictionary[k]
        else:
            print(k + ' is an unknown argument!')
            assert(False)
    for k in new_dictionary.keys():
        if k[0] == '-':
            new_dictionary[k[1:]] = new_dictionary[k]
            del new_dictionary[k]
    return new_dictionary


def text_to_file(d):
    txt = d['txt']
    path = d['path']

    with open(path, "w") as text_file:
        text_file.write("{0}".format(txt))


def img_to_img_uint8(d):
    img = d['img']

    return (255.0 * z2o(img)).astype(np.uint8)


def zsave_obj(d):
    obj = d['obj']
    path = d['path']
    if 'save_function_placeholder' not in d:
        save_function_placeholder = True

    if path is not None:
        print(path)
    if callable(obj):
        if save_function_placeholder:
            text_to_file({'txt': '<function>', 'path': path + '.fun'})
        else:
            pass
    elif isinstance(obj, str):
        text_to_file({'txt': obj, 'path': path + '.txt'})
    elif fname(path) == 'img_uint8':
        imsave(path + '.png', obj)
    elif isinstance(obj, dict):
        assert(path is not None)
        unix('mkdir -p ' + path)
        for k in obj.keys():
            zsave_obj({'obj': obj[k], 'path': opj(path, k)})
    else:
        save_obj(obj, path)


def zload_obj(d):
    path = d['path']

    if 'ctr' not in d:
        ctr = 0
    else:
        ctr = d['ctr']

    print(path, ctr)
    obj = {}
    txt = sggo(path, '*.txt')
    fun = sggo(path, '*.fun')
    pkl = sggo(path, '*.pkl')
    img_uint8 = sggo(path, '*.png')
    all_files = sggo(path, '*')
    dic = []
    for a in all_files:
        if os.path.isdir(a):
            dic.append(a)
    # print(dic)
    # print txt
    # print fun
    # print pkl
    # print img_uint8
    # print dic
    #raw_input('hit enter')
    for k in txt:
        q = '\n'.join(txt_file_to_list_of_strings(k))
        n = fname(k).split('.')[0]
        obj[n] = q
    for k in fun:
        n = fname(k).split('.')[0]
        #print('do nothing with '+k)
        #obj[n] = '<function>'
    for k in pkl:
        n = fname(k).split('.')[0]
        obj[n] = load_obj(k)
    for k in img_uint8:
        n = fname(k).split('.')[0]
        obj[n] = imread(k)
    for k in dic:
        n = fname(k)
        # print(dic,n,k,ctr)
        obj[n] = zload_obj({'path': k, 'ctr': ctr + 1})

    #raw_input('hit enter')
    return obj


def restore_functions(d):
    src = d['src']
    dst = d['dst']

    for k in src.keys():
        if callable(src[k]):
            dst[k] = src[k]
        elif isinstance(src[k], dict):
            restore_functions({'src': src[k], 'dst': dst[k]})
        else:
            pass


def array_to_int_list(a):
    l = []
    for d in a:
        l.append(int(d * 100))
    return l


#c = code_to_code_str({'path':path, 'start':106   })

def code_to_code_str(d):
    import pyperclip
    path = d['path']

    code = txt_file_to_list_of_strings(path)
    for i in range(len(code)):
        pd2s(i, ')', code[i])
    start, stop = input('start,stop ')
    code_to_clipboard({'path': path, 'start': start, 'stop': stop})


def code_to_clipboard(d):
    import pyperclip
    code = d['path']
    start = d['start']
    stop = d['stop']

    code_str = '\n'.join(code[start:stop])
    cprint(code_str, 'yellow')
    pyperclip.copy(code_str)
    print('\nOkay, it is in the clipboard')


def blank_dic():
    print("""
def blank_dic(d):
    D = {}
    D[''] = d['']
    True
    D['type'] = '?'
    D['Purpose'] = d2s(inspect.stack()[0][3],':','?')
    return D""")


def blank_file():
    print("""
from kzpy3.utils2 import *
pythonpaths(['kzpy3'])
from vis2 import *


translation_dic = {'a':'apples','b':'build','c':'cats','d':'dogs'}
if __name__ == "__main__" and '__file__' in vars():
    argument_dictionary = args_to_dic({  'pargs':sys.argv[1:]  })
else:
    print('Running this within interactive python.')
    argument_dictionary = args_to_dic({  'pargs':"-a -1 -b 4 -c [1,2,9] -d {1:5,2:4}"  })
argument_dictionary = translate_args(
    {'argument_dictionary':argument_dictionary,
    'translation_dic':translation_dic})
print(argument_dictionary)


        """)
    blank_dic()
    print("""

if False:
    try:
        pass
    except Exception as e:
        print("********** Exception ***********************")
        print(e.message, e.args)

#EOF
    """)


def array_to_int_list(a):
    l = []
    for d in a:
        l.append(int(d * 100))
    return l


def text_to_file(d):
    txt = d['txt']
    path = d['path']
    True
    with open(path, "w") as text_file:
        text_file.write("{0}".format(txt))


def img_to_img_uint8(d):
    img = d['img']
    True
    return (255.0 * z2o(img)).astype(np.uint8)


def zsave_obj(d):
    obj = d['obj']
    path = d['path']
    True
    if path is not None:
        print(path)
    if callable(obj):
        text_to_file({'txt': '<function>', 'path': path + '.fun'})
    elif isinstance(obj, str):
        text_to_file({'txt': obj, 'path': path + '.txt'})
    elif fname(path) == 'img_uint8':
        imsave(path + '.png', obj)
    elif isinstance(obj, dict):
        assert(path is not None)
        unix('mkdir -p ' + path)
        for k in obj.keys():
            zsave_obj({'obj': obj[k], 'path': opj(path, k)})
    else:
        save_obj(obj, path)


def zload_obj(d):
    path = d['path']
    True
    if 'ctr' not in d:
        ctr = 0
    else:
        ctr = d['ctr']

    print(path, ctr)
    obj = {}
    txt = sggo(path, '*.txt')
    fun = sggo(path, '*.fun')
    pkl = sggo(path, '*.pkl')
    img_uint8 = sggo(path, '*.png')
    all_files = sggo(path, '*')
    dic = []
    for a in all_files:
        if os.path.isdir(a):
            dic.append(a)
    print(dic)
    # print txt
    # print fun
    # print pkl
    # print img_uint8
    # print dic
    #raw_input('hit enter')
    for k in txt:
        q = '\n'.join(txt_file_to_list_of_strings(k))
        n = fname(k).split('.')[0]
        obj[n] = q
    for k in fun:
        n = fname(k).split('.')[0]
        print('do nothing with ' + k)
        #obj[n] = '<function>'
    for k in pkl:
        n = fname(k).split('.')[0]
        obj[n] = load_obj(k)
    for k in img_uint8:
        n = fname(k).split('.')[0]
        obj[n] = imread(k)
    for k in dic:
        n = fname(k)
        print(dic, n, k, ctr)
        obj[n] = zload_obj({'path': k, 'ctr': ctr + 1})

    #raw_input('hit enter')
    return obj


def zrestore_functions(d):
    src = d['src']
    dst = d['dst']
    True
    for k in src.keys():
        if callable(src[k]):
            dst[k] = src[k]
        elif isinstance(src[k], dict):
            restore_functions({'src': src[k], 'dst': dst[k]})
        else:
            pass


def stop_ros():
    #M['Stop_Arduinos'] = True
    #rospy.signal_shutdown("M[Stop_Arduinos] = True")
    print('!!!!! stop_ros() !!!!!')
    # time.sleep(1)
    unix(opjh('kzpy3/kill_ros.sh'))
    # assert(False)


# EOF
