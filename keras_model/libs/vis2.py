"""Visualization utilities imported from kzpy3."""
import sys
import traceback
from utils2 import *

import matplotlib
try:
    import cv2
except BaseException:
    print("Couldn't import cv2")


MacOSX = False
if '/Users/' in home_path:
    MacOSX = True

if MacOSX:
    matplotlib.use(u'MacOSX')


###########
'''
e.g.
from vis import *; model_car_vis_test()
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
PP, FF = plt.rcParams, 'figure.figsize'


def hist(data, bins=100):
    """
    default hist behavior
    """
    plt.clf()
    plt.hist(data, bins=bins)
    pass


plot = plt.plot
figure = plt.figure
clf = plt.clf


def toolbar():
    plt.rcParams['toolbar'] = 'toolbar2'
    
######################



def mi(
        image_matrix,
        figure_num=1,
        subplot_array=[1, 1, 1],
        img_title='',
        img_xlabel='x',
        img_ylabel='y',
        cmap='gray',
        toolBar=True,
        do_clf=True,
        do_axis=False):
    """
    My Imagesc, displays a matrix as grayscale image if 2d, or color if 3d.
    Can take different inputs -- e.g.,

        from matrix:

            from vis import *
            mi(np.random.rand(256,256),99,[1,1,1],'random matrix')

        from path:
            mi(opjh('Desktop','conv1'),1,[5,5,0])

        from list:
            l = load_img_folder_to_list(opjh('Desktop','conv5'))
            mi(l,2,[4,3,0])

        from dict:
            mi(load_img_folder_to_dict(opjh('Desktop','conv5')),1,[3,4,0])
    """
    if isinstance(image_matrix, str):
        l = load_img_folder_to_list(image_matrix)
        mi(l)
        return

    if isinstance(image_matrix, list):
        l = 1.0 * array(image_matrix)
        l /= l.max()
        mi(vis_square(l))
        return

    if isinstance(image_matrix, dict):
        img_keys = sorted(image_matrix.keys(), key=natural_keys)
        l = []
        for k in img_keys:
            l.append(image_matrix[k])
        mi(l)
        return

    if not toolBar:
        plt.rcParams['toolbar'] = 'None'
    else:
        plt.rcParams['toolbar'] = 'toolbar2'

    f = plt.figure(figure_num)
    if do_clf:
        # print('plt.clf()')
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
    f.add_subplot(subplot_array[0], subplot_array[1], subplot_array[2])
    imgplot = plt.imshow(image_matrix, cmap)
    imgplot.set_interpolation('nearest')
    if not do_axis:
        plt.axis('off')
    if len(img_title) > 0:  # != 'no title':
        plt.title(img_title)

######################


def load_img_folder_to_dict(img_folder):
    '''Assume that *.* selects only images.'''
    img_fns = gg(opj(img_folder, '*.*'))
    imgs = {}
    for f in img_fns:
        imgs[fname(f)] = imread(f)
    return imgs


def load_img_folder_to_list(img_folder):
    return dict_to_sorted_list(load_img_folder_to_dict(img_folder))


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx.
# sqrt(n) by sqrt(n)
def vis_square(data_in, padsize=1, padval=0):
    data = data_in.copy()
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize),
               (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(
        data,
        padding,
        mode='constant',
        constant_values=(
            padval,
            padval))

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data


def apply_rect_to_img(
        img,
        value,
        min_val,
        max_val,
        pos_color,
        neg_color,
        rel_bar_height,
        rel_bar_thickness,
        center=False,
        reverse=False,
        horizontal=False):
    # print(value)
    h, w, d = shape(img)
    p = (value - min_val) / (max_val - 1.0 * min_val)
    if reverse:
        p = 1.0 - p
    if p > 1:
        p = 1
    if p < 0:
        p = 0
    wp = int(p * w)
    hp = int(p * h)
    bh = int((1 - rel_bar_height) * h)
    bt = int(rel_bar_thickness * h)
    bw = int((1 - rel_bar_height) * w)

    if horizontal:
        if center:
            if wp < w / 2:
                img[(bh - bt / 2):(bh + bt / 2), (wp):(w / 2), :] = neg_color
            else:
                img[(bh - bt / 2):(bh + bt / 2), (w / 2):(wp), :] = pos_color
        else:
            img[(bh - bt / 2):(bh + bt / 2), 0:wp, :] = pos_color
    else:
        if center:
            if hp < h / 2:
                img[(hp):(h / 2), (bw - bt / 2):(bw + bt / 2), :] = neg_color
            else:
                img[(h / 2):(hp), (bw - bt / 2):(bw + bt / 2), :] = pos_color

        else:
            img[hp:h, (bw - bt / 2):(bw + bt / 2), :] = pos_color


def plt_square():
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()


def function_close_all_windows():
    plt.close('all')


CA = function_close_all_windows


def mci(
        img,
        delay=33,
        title='animate',
        scale=1.0,
        color_mode=cv2.COLOR_RGB2BGR):
    img = cv2.cvtColor(img, color_mode)
    scale_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(title, scale_img)
    k = cv2.waitKey(delay)
    return k


def mcia(
        img_block,
        delay=33,
        title='animate',
        scale=1.0,
        color_mode=cv2.COLOR_RGB2BGR):
    assert(len(shape(img_block)) == 4)
    for i in range(shape(img_block)[0]):
        k = mci(img_block[i, :, :, :], delay=delay,
                title=title, scale=scale, color_mode=color_mode)
        if k == ord('q'):
            return


def mcia_folder(
        path,
        delay=33,
        title='animate',
        scale=1.0,
        color_mode=cv2.COLOR_RGB2BGR):
    l = load_img_folder_to_list(path)
    mcia(
        array(l),
        delay=delay,
        title=title,
        scale=scale,
        color_mode=cv2.COLOR_RGB2BGR)


def frames_to_video_with_ffmpeg(input_dir, output_path, img_range=(), rate=30):
    if input_dir[-1] == '/':
        input_dir = input_dir[:-1]  # the trailing / messes up the name.
    _, fnames = dir_as_dic_and_list(input_dir)
    frames_folder = input_dir.split('/')[-1]
    unix('mkdir -p ' + '/'.join(output_path.split('/')[:-1]))
    unix_str = ' -i ' + input_dir + '/%d.png -pix_fmt yuv420p -r ' + \
        str(rate) + ' -b:v 14000k ' + output_path
    success = False
    try:
        print('Trying avconv.')
        unix('avconv' + unix_str)
        success = True
    except Exception as e:
        print("'avconv did not work.' ***************************************")
        print(e.message, e.args)
        print("***************************************")
    if not success:
        try:
            print('Trying ffmpeg.')
            unix('ffmpeg' + unix_str)
            success = True
        except Exception as e:
            print("'ffmeg did not work.' ***************************************")
            print(e.message, e.args)
            print("***************************************")
    if success:
        print('frames_to_video_with_ffmpeg() had success with ' + frames_folder)


def iadd(src, dst, xy, neg=False):
    src_size = []
    upper_corner = []
    lower_corner = []
    for i in [0, 1]:
        src_size.append(shape(src)[i])
        upper_corner.append(int(xy[i] - src_size[i] / 2.0))
        lower_corner.append(int(xy[i] + src_size[i] / 2.0))
    if neg:
        dst[upper_corner[0]:lower_corner[0],
            upper_corner[1]:lower_corner[1]] -= src
    else:
        dst[upper_corner[0]:lower_corner[0],
            upper_corner[1]:lower_corner[1]] += src


def isub(src, dst, xy):
    iadd(src, dst, xy, neg=True)


def pt_plot(xy, color='r'):
    plot(xy[0], xy[1], color + '.')


def pts_plot(xys, color='r'):
    assert(len(color) == 1)
    x = xys[:, 0]
    y = xys[:, 1]
    plot(x, y, color + '.')


###########
#
def Image(xyz_sizes, origin, mult, data_type=np.uint8):
    D = {}
    D['origin'] = origin
    D['mult'] = mult
    D['Purpose'] = d2s(
        inspect.stack()[0][3],
        ':',
        'An image which translates from float coordinates.')
    D['name'] = 'Image'

    def _floats_to_pixels(xy):
        xy = array(xy)
        xyn = 0 * xy
        if len(shape(xy)) == 1:
            xyn[0] = D['mult'] * xy[0]
            xyn[0] += D['origin']
            xyn[1] = D['mult'] * xy[1]
            xyn[1] += D['origin']
        else:
            xyn[:, 0] = D['mult'] * xy[:, 0]
            xyn[:, 0] += D['origin']
            xyn[:, 1] = D['mult'] * xy[:, 1]
            xyn[:, 1] += D['origin']
        return np.ndarray.astype(xyn, int)

    def _pixel_to_float(xy):
        xy = array(xy)
        xyn = 0.0 * xy
        assert(len(shape(xy)) == 1)
        xyn[0] = xy[0] - D['origin']
        xyn[0] /= (1.0 * D['mult'])
        xyn[1] = xy[1] - D['origin']
        xyn[1] /= (1.0 * D['mult'])
        return np.ndarray.astype(xyn, float)
    D['floats_to_pixels'] = _floats_to_pixels
    D['pixel_to_float'] = _pixel_to_float

    def _pts_plot(xy, c='b'):
        if len(xy) < 1:
            #print('warning, asked to plot empty pts')
            return
        xy_pix = D['floats_to_pixels'](xy)
        if len(shape(xy)) == 1:
            plot(xy_pix[1], xy_pix[0], c + '.')
        else:
            plot(xy_pix[:, 1], xy_pix[:, 0], c + '.')
    D['pts_plot'] = _pts_plot

    def _apply_fun(f):
        for x in range(0, 2 * D['origin']):
            for y in range(0, 2 * D['origin']):
                xy_float = D['pixel_to_float']((x, y))
                D['img'][x][y] = f(xy_float[0], xy_float[1])
    D['apply_fun'] = _apply_fun

    def _show(name=None):
        if name is None:
            name = D['name']
        mi(D['img'], name)
        # prin(t d2s('name =',name))
    D['show'] = _show

    def _clear():
        D['img'] *= 0.0
    if len(xyz_sizes) == 2:
        D['img'] = zeros((xyz_sizes[0], xyz_sizes[1]), data_type)
    elif len(xyz_sizes) == 3:
        D['img'] = zeros((xyz_sizes[0], xyz_sizes[1], xyz_sizes[2]), data_type)
    else:
        assert(False)
    return D


#
###############


###########
# https://stackoverflow.com/questions/35281427/fast-python-plotting-library-to-draw-plots-directly-on-2d-numpy-array-image-buff
def Plot(xy_pix_sizes, origin, xy_mults):
    D = {}
    D['origin'] = origin
    D['xy_mults'] = xy_mults
    D['Purpose'] = d2s(inspect.stack()[0][3], ':', 'A cv2 ploter.')
    D['name'] = 'Image'

    def _floats_to_pixels(xy):
        xy = array(xy)
        xyn = 0 * xy
        if len(shape(xy)) == 1:
            xyn[0] = D['mult'] * xy[0]
            xyn[0] += D['origin']
            xyn[1] = D['mult'] * xy[1]
            xyn[1] += D['origin']
        else:
            xyn[:, 0] = D['mult'] * xy[:, 0]
            xyn[:, 0] += D['origin']
            xyn[:, 1] = D['mult'] * xy[:, 1]
            xyn[:, 1] += D['origin']
        return np.ndarray.astype(xyn, int)

    def _pixel_to_float(xy):
        xy = array(xy)
        xyn = 0.0 * xy
        assert(len(shape(xy)) == 1)
        xyn[0] = xy[0] - D['origin']
        xyn[0] /= (1.0 * D['mult'])
        xyn[1] = xy[1] - D['origin']
        xyn[1] /= (1.0 * D['mult'])
        return np.ndarray.astype(xyn, float)
    D['floats_to_pixels'] = _floats_to_pixels
    D['pixel_to_float'] = _pixel_to_float

    def _pts_plot(xy, c='b'):
        if len(xy) < 1:
            print('warning, asked to plot empty pts')
            return
        xy_pix = D['floats_to_pixels'](xy)
        if len(shape(xy)) == 1:
            pass  # plot(xy_pix[1],xy_pix[0],c+'.')
        else:
            pass  # plot(xy_pix[:,1],xy_pix[:,0],c+'.')
    D['pts_plot'] = _pts_plot

    def _apply_fun(f):
        for x in range(0, 2 * D['origin']):
            for y in range(0, 2 * D['origin']):
                xy_float = D['pixel_to_float']((x, y))
                D['img'][x][y] = f(xy_float[0], xy_float[1])    # ???
    D['apply_fun'] = _apply_fun

    def _show(name=None):
        if name is None:
            name = D['name']
        mi(D['img'], name)
        # prin(t d2s('name =',name))
    D['show'] = _show

    def _clear():
        D['img'] *= 0.0
    if len(xyz_sizes) == 2:
        D['img'] = zeros((xyz_sizes[0], xyz_sizes[1]), data_type)
    elif len(xyz_sizes) == 3:
        D['img'] = zeros((xyz_sizes[0], xyz_sizes[1], xyz_sizes[2]), data_type)
    else:
        assert(False)
    return D
#
###############


def xylim(a, b, c, d):
    xlim(a, b)
    ylim(c, d)



# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
from math import acos
from math import sqrt
from math import pi


def length(v):
    return math.sqrt(v[0]**2 + v[1]**2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w) / (length(v) * length(w))
    if cosx > 1.0:
        cosx = 1.0
    elif cosx < -1.0:
        cosx = -1.0
    rad = acos(cosx)  # in radians
    return rad * 180 / pi  # returns degrees


def angle_clockwise(A, B):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360 - inner


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


def rotatePoint(centerPoint, point, angle):
    """http://stackoverflow.com/questions/20023209/function-for-rotating-2d-objects
    Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0] - centerPoint[0], point[1] - centerPoint[1]
    temp_point = (
        temp_point[0] *
        math.cos(angle) -
        temp_point[1] *
        math.sin(angle),
        temp_point[0] *
        math.sin(angle) +
        temp_point[1] *
        math.cos(angle))
    temp_point = temp_point[0] + centerPoint[0], temp_point[1] + centerPoint[1]
    return temp_point


def rotatePolygon(polygon, theta):
    """http://stackoverflow.com/questions/20023209/function-for-rotating-2d-objects
    Rotates the given polygon which consists of corners represented as (x,y),
    around the ORIGIN, clock-wise, theta degrees"""
    theta = math.radians(theta)
    rotatedPolygon = []
    for corner in polygon:
        rotatedPolygon.append(
            (corner[0] *
             math.cos(theta) -
             corner[1] *
                math.sin(theta),
                corner[0] *
                math.sin(theta) +
                corner[1] *
                math.cos(theta)))
    return rotatedPolygon


def length(xy):
    return sqrt(xy[0]**2 + xy[1]**2)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    http://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python

    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)


def Gaussian_2D(width):
    return makeGaussian(width, width / 3.0)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def f___(x, A, B):
    return A * x + B


def normalized_vector_from_pts(pts):
    pts = array(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    m, b = curve_fit(f___, x, y)[0]
    heading = normalized([1, m])[0]
    len_heading = length(heading)
    # print len_heading
    # if np.abs(len_heading-1.0)>0.1:
    #    print('here')
    #    print((heading,len_heading,pts))
    #    assert(False)
    return heading


def Image_to_Folder_Saver(d):
    D = {}
    D['path'] = d['path']

    D['type'] = 'Image_to_Folder_Saver'
    D['Purpose'] = d2s(
        inspect.stack()[0][3],
        ':',
        'Save images to folder with counter for name.')
    D['save_img_ctr'] = 0

    def _save(d):
        img = d['img']
        if 'ext' not in d:
            ext = 'png'
        imsave(opj(D['path'], str(D['save_img_ctr']) + '.' + ext), img)
        D['save_img_ctr'] += 1
    D['save'] = _save
    return D
