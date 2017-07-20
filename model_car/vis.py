from model_car.utils import *
import sys, traceback
import matplotlib
try:
    import cv2
except:
    print("Couldn't import cv2")


MacOSX = False
if '/Users/' in home_path:
    MacOSX = True

if MacOSX:
    matplotlib.use(u'MacOSX')


###########
'''
e.g.
from model_car.vis import *; model_car_vis_test()
'''
################

import matplotlib.pyplot as plt  # the Python plotting package
plt.ion()
plot = plt.plot
hist = plt.hist
xlim = plt.xlim
ylim = plt.ylim
clf = plt.clf
pause = plt.pause
figure = plt.figure
title = plt.title
plt.ion()
plt.show()
PP,FF = plt.rcParams,'figure.figsize'

#plt.figure(figsize=(6.5, 4))

def model_car_vis_test():
    img_dic = get_some_images()
    ppff = PP[FF]
    PP[FF] = 3,3
    mi(img_dic['bay'],'bay')
    PP[FF] = ppff
    plt.figure('hist')
    plt.hist(np.random.randn(10000),bins=100)
    True

def hist(data,bins=100):
    """
    default hist behavior
    """
    plt.clf()
    plt.hist(data,bins=bins)
    pass
plot = plt.plot
figure = plt.figure
clf=plt.clf



try:
    # - These allow for real-time display updating
    from cStringIO import StringIO
    import scipy.ndimage as nd
    import PIL.Image
    if MacOSX:
        from IPython.display import clear_output, Image, display
    def showarray(a, fmt='jpeg'):
        a = np.uint8(np.clip(255.0*z2o(a), 0, 255))
        f = StringIO()
        PIL.Image.fromarray(a).save(f, fmt)
        display(Image(data=f.getvalue()))
except:
    print("model_car.vis: PIL image display not imported.")

def toolbar():
    plt.rcParams['toolbar'] = 'toolbar2'
    
######################
#
def mi(
    image_matrix,
    figure_num = 1,
    subplot_array = [1,1,1],
    img_title = '',
    img_xlabel = 'x',
    img_ylabel = 'y',
    cmap = 'gray',
    toolBar = True,
    do_clf = True,
    do_axis = False ):
    """
    My Imagesc, displays a matrix as grayscale image if 2d, or color if 3d.
    Can take different inputs -- e.g.,

        from matrix:

            from model_car.vis import *
            mi(np.random.rand(256,256),99,[1,1,1],'random matrix')

        from path:
            mi(opjh('Desktop','conv1'),1,[5,5,0])

        from list:
            l = load_img_folder_to_list(opjh('Desktop','conv5'))
            mi(l,2,[4,3,0])

        from dict:
            mi(load_img_folder_to_dict(opjh('Desktop','conv5')),1,[3,4,0])
    """
    if type(image_matrix) == str:
        mi(load_img_folder_to_dict(image_matrix),image_matrix,subplot_array,img_title,img_xlabel,img_ylabel,cmap,toolBar)
        return

    if type(image_matrix) == list:
        if np.array(subplot_array).max() < 2:
            subplot_array = [1,len(image_matrix),0]
        for i in range(len(image_matrix)):
            mi(image_matrix[i],figure_num,[subplot_array[0],subplot_array[1],i+1],img_title,img_xlabel,img_ylabel,cmap,toolBar)
        return

    if type(image_matrix) == dict:
        if np.array(subplot_array).max() < 2:
            subplot_array = [1,len(image_matrix),0]
        i = 0
        img_keys = sorted(image_matrix.keys(),key=natural_keys)
        for k in img_keys:
            mi(image_matrix[k],figure_num,[subplot_array[0],subplot_array[1],i+1],img_title,img_xlabel,img_ylabel,cmap,toolBar)
            i += 1
        return        

    if toolBar == False:
        plt.rcParams['toolbar'] = 'None'
    else:
        plt.rcParams['toolbar'] = 'toolbar2'

    f = plt.figure(figure_num)
    if do_clf:
        #print('plt.clf()')
        plt.clf()

    if True:
        f.subplots_adjust(bottom=0.05)
        f.subplots_adjust(top=0.95)
        f.subplots_adjust(wspace=0.1)
        f.subplots_adjust(hspace=0.1)
        f.subplots_adjust(left=0.05)
        f.subplots_adjust(right=0.95)
    if False:
        f.subplots_adjust(bottom=0.0)
        f.subplots_adjust(top=0.95)
        f.subplots_adjust(wspace=0.0)
        f.subplots_adjust(hspace=0.1)
        f.subplots_adjust(left=0.0)
        f.subplots_adjust(right=1.0)
    f.add_subplot(subplot_array[0],subplot_array[1],subplot_array[2])
    imgplot = plt.imshow(image_matrix, cmap)
    imgplot.set_interpolation('nearest')
    if not do_axis:
        plt.axis('off')
    if len(img_title) > 0:# != 'no title':
        plt.title(img_title)
#
######################









def mp(args,figure_num=1, subplot_array=[1,1,1],
       title='', xlabel='', ylabel='', xlim=[], ylim=[], toolBar=False):

    if toolBar == False:
        plt.rcParams['toolbar'] = 'None'
    else:
        plt.rcParams['toolbar'] = 'toolbar2'

    f = plt.figure(figure_num)

    if False:
        f.subplots_adjust(bottom=0.05)
        f.subplots_adjust(top=0.95)
        f.subplots_adjust(wspace=0.1)
        f.subplots_adjust(hspace=0.1)
        f.subplots_adjust(left=0.05)
        f.subplots_adjust(right=0.95)

    f.add_subplot(subplot_array[0],subplot_array[1],subplot_array[2])
    imgplot = plt.plot(*args)
    if len(title) > 0:# != 'no title':
        plt.title(title)
    else:
        plt.title(str(subplot_array[2]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(xlim)==2:
        plt.xlim(xlim)    
    if len(ylim)==2:
        plt.ylim(ylim)


def yb_color_modulation_of_grayscale_image(img,y,b,opt_lower_contrast=True):

    if len(np.shape(img))>2:
        img = np.mean(img,axis=2)
    img = z2o(img)

    if opt_lower_contrast:
        print('low contrast option')
        img = (1.0+img)/3.0

    y = z2o(y)
    b = z2o(b)

    ci = np.zeros((np.shape(img)[0],np.shape(img)[1],3))
    print(np.shape(ci))
    for i in range(3):
        ci[:,:,i] = 1.0*img
    ci = ci/np.max(ci)

    for i in range(3):
        ci[:,:,i] *= (1-y)
    for i in [0,1]:
        ci[:,:,i] += y

    for i in range(3):
        ci[:,:,i] *= (1-b)
    for i in [2]:
        ci[:,:,i] += b
        
    return ci

    
 
def get_some_images():
    '''
    Load some images that can be used for demos, etc.
    e.g., img_dic = get_some_images(); mi(img_dic['bay'])
    '''
    img_dic = {}
    img_dic['bay'] = imread(opj(home_path,'Pictures','bay2.png'))
    return img_dic



# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data_in, padsize=1, padval=0):
    data = data_in.copy()
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    return data



import matplotlib.colors
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)

''' from http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
e.g.,

c = matplotlib.colors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))
plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
plt.colorbar()
plt.show()
'''





def load_img_folder_to_dict(img_folder):
    '''Assume that *.* selects only images.'''
    img_fns = gg(opj(img_folder,'*.*'))
    imgs = {}
    for f in img_fns:
        imgs[f.split('/')[-1]] = imread(f)
    return imgs

def load_img_folder_to_list(img_folder):
    return dict_to_sorted_list(load_img_folder_to_dict(img_folder))



def my_scatter(x,y,xmin,xmax,fig_wid,fig_name):
    plt.figure(fig_name,(fig_wid,fig_wid))
    plt.clf()
    plt.plot(x,y,'bo')
    plt.title(np.corrcoef(x,y)[0,1])
    plt.xlim(xmin,xmax)
    plt.ylim(xmin,xmax)



def apply_rect_to_img(img,value,min_val,max_val,pos_color,neg_color,rel_bar_height,rel_bar_thickness,center=False,reverse=False,horizontal=False):
    #print(value)
    h,w,d = shape(img)
    p = (value - min_val) / (max_val - 1.0*min_val)
    if reverse:
        p = 1.0 - p
    if p > 1:
        p = 1
    if p < 0:
        p = 0
    wp = int(p*w)
    hp = int(p*h)
    bh = int((1-rel_bar_height) * h)
    bt = int(rel_bar_thickness * h)
    bw = int((1-rel_bar_height) * w)

    if horizontal:
        if center:
            if wp < w/2:
                img[(bh-bt/2):(bh+bt/2),(wp):(w/2),:] = neg_color
            else:
                img[(bh-bt/2):(bh+bt/2),(w/2):(wp),:] = pos_color
        else:
            img[(bh-bt/2):(bh+bt/2),0:wp,:] = pos_color
    else:
        if center:
            if hp < h/2:
                img[(hp):(h/2),(bw-bt/2):(bw+bt/2),:] = neg_color
            else:
                img[(h/2):(hp),(bw-bt/2):(bw+bt/2),:] = pos_color

        else:
            img[hp:h,(bw-bt/2):(bw+bt/2),:] = pos_color


def plt_square():
    plt.gca().set_aspect('equal',adjustable='box')
    plt.draw()



def function_close_all_windows():
    plt.close('all')
CA = function_close_all_windows



def mi_or_cv2_animate(img_array,cv=True,delay=30,title='animate'):
    if type(img_array)==np.ndarray:
        for i in range(len(img_array)):
             mi_or_cv2(img_array[i],cv,delay,title)        
    elif type(img_array)==np.ndarray:
        for i in range(len(img_array[0])):
             mi_or_cv2(img_array[i],cv,delay,title)
    else:
        print('I am confused')
        assert(False)


def mci(img,delay=33,title='animate',scale=1.0,color_mode=cv2.COLOR_RGB2BGR):
    img = cv2.cvtColor(img,color_mode)
    scale_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    cv2.imshow(title,scale_img)
    k = cv2.waitKey(delay)
    return k

def mcia(img_block,delay=33,title='animate',scale=1.0,color_mode=cv2.COLOR_RGB2BGR):
    assert(len(shape(img_block)) == 4)
    for i in range(shape(img_block)[0]):
        k = mci(img_block[i,:,:,:],delay,title,scale,color_mode)
        if k == ord('q'):
            return

def mi_or_cv2(img,cv=True,delay=30,title='animate'):
    if cv:
        cv2.imshow(title,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            pass
    else:
        mi(img,title)
        pause(0.0001)




def frames_to_video_with_ffmpeg(input_dir,output_path,img_range=(),rate=30):
    if input_dir[-1] == '/':
        input_dir = input_dir[:-1] # the trailing / messes up the name.
    _,fnames = dir_as_dic_and_list(input_dir)
    frames_folder = input_dir.split('/')[-1]
    unix('mkdir -p '+'/'.join(output_path.split('/')[:-1]))
    unix_str = ' -i '+input_dir+'/%d.png -pix_fmt yuv420p -r '+str(rate)+' -b:v 14000k '+output_path
    success = False
    try:
        print('Trying avconv.')
        unix('avconv'+unix_str)
        success = True
    except Exception as e:
        print "'avconv did not work.' ***************************************"
        traceback.print_exc(file=sys.stdout)
        #print e.message, e.args
        print "***************************************"
    if not success:
        try:
            print('Trying ffmpeg.')
            unix('ffmpeg'+unix_str)
            success = True
        except Exception as e:
            print "'ffmeg did not work.' ***************************************"
            traceback.print_exc(file=sys.stdout)
            #print e.message, e.args
            print "***************************************"
    if success:
        print('frames_to_video_with_ffmpeg() had success with ' + frames_folder)








def iadd(src,dst,xy,neg=False):
    src_size = []
    upper_corner = []
    lower_corner = []
    for i in [0,1]:
        src_size.append(shape(src)[i])
        upper_corner.append(int(xy[i]-src_size[i]/2.0))
        lower_corner.append(int(xy[i]+src_size[i]/2.0))
    if neg:
        dst[upper_corner[0]:lower_corner[0],upper_corner[1]:lower_corner[1]] -= src
    else:
        dst[upper_corner[0]:lower_corner[0],upper_corner[1]:lower_corner[1]] += src
    
def isub(src,dst,xy):
    iadd(src,dst,xy,neg=True)





def pt_plot(xy,color='r'):
    plot(xy[0],xy[1],color+'.')

def pts_plot(xys,color='r'):
    assert(len(color)==1)
    x = xys[:,0]
    y = xys[:,1]
    plot(x,y,color+'.')

        

###########
#
def Image(xyz_sizes,origin,mult,data_type=np.uint8):
    D = {}
    D['origin'] = origin
    D['mult'] = mult
    D['Purpose'] = 'An image which translates from float coordinates.'
    def _floats_to_pixels(xy):
        """
        xy = array(xy)
        if len(shape(xy)) == 1:
            xy[0] *= -D['mult']
            xy[0] += D['origin']
            xy[1] *= D['mult']
            xy[1] += D['origin']
        else:
            xy[:,0] *= -D['mult']
            xy[:,0] += D['origin']
            xy[:,1] *= D['mult']
            xy[:,1] += D['origin']
        """
        xy = array(xy)
        xyn = 0*xy
        if len(shape(xy)) == 1:
            xyn[0] = D['mult'] * xy[0]
            xyn[0] += D['origin']
            xyn[1] = D['mult'] * xy[1]
            xyn[1] += D['origin']
        else:
            xyn[:,0] = D['mult'] * xy[:,0]
            xyn[:,0] += D['origin']
            xyn[:,1] = D['mult'] * xy[:,1]
            xyn[:,1] += D['origin']
        return np.ndarray.astype(xyn,int)
    def _pixel_to_float(xy):
        xy = array(xy)
        xyn = 0.0*xy
        assert(len(shape(xy)) == 1)
        xyn[0] = xy[0] - D['origin']
        xyn[0] /= (1.0*D['mult'])
        xyn[1] = xy[1] - D['origin']
        xyn[1] /= (1.0*D['mult'])
        return np.ndarray.astype(xyn,float)
    D['floats_to_pixels'] = _floats_to_pixels
    D['pixel_to_float'] = _pixel_to_float
    def _plot_pts(xy,c='b'):
        if len(xy) < 1:
            #print('warning, asked to plot empty pts')
            return
        xy_pix = D['floats_to_pixels'](xy)
        if len(shape(xy)) == 1:
            plot(xy_pix[1],xy_pix[0],c+'.')
        else:
            plot(xy_pix[:,1],xy_pix[:,0],c+'.')
    D['plot_pts'] = _plot_pts
    if len(xyz_sizes) == 2:
        D['img'] = zeros((xyz_sizes[0],xyz_sizes[1]),data_type)
    elif len(xyz_sizes) == 3:
        D['img'] = zeros((xyz_sizes[0],xyz_sizes[1],xyz_sizes[2]),data_type)
    else:
        assert(False)
    return D


#
###############

def xylim(a,b,c,d):
    xlim(a,b)
    ylim(c,d)




# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
from math import acos
from math import sqrt
from math import pi
def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner




def unit_vector(vector):
    """http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """http://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def rotatePoint(centerPoint,point,angle):
    """http://stackoverflow.com/questions/20023209/function-for-rotating-2d-objects
    Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point



def rotatePolygon(polygon,theta):
    """http://stackoverflow.com/questions/20023209/function-for-rotating-2d-objects
    Rotates the given polygon which consists of corners represented as (x,y),
    around the ORIGIN, clock-wise, theta degrees"""
    theta = math.radians(theta)
    rotatedPolygon = []
    for corner in polygon :
        rotatedPolygon.append(( corner[0]*math.cos(theta)-corner[1]*math.sin(theta) , corner[0]*math.sin(theta)+corner[1]*math.cos(theta)) )
    return rotatedPolygon


            


def length(xy):
    return sqrt(xy[0]**2+xy[1]**2)






def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    http://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python

    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def Gaussian_2D(width):
    return makeGaussian(width,width/3.0)



def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)



def f(x,A,B):
    return A*x+B

def normalized_vector_from_pts(pts):
    pts = array(pts)
    x = pts[:,0]
    y = pts[:,1]
    m,b = curve_fit(f,x,y)[0]
    heading = normalized([1,m])[0]
    return heading



