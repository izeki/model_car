from model_car.utils import *
import rospy
import rosbag
import cv2
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError
bridge = cv_bridge.CvBridge()
from model_car.data_analysis.data_parsing.Bagfile_Handler import Bagfile_Handler


def load_images(bag_file_path,color_mode="rgb8",include_flip=True):
    
    topic_name_map = {}
    
    topic_name_map['/bair_car/zed/left/image_rect_color'] = 'left'
    topic_name_map['/bair_car/zed/right/image_rect_color'] = 'right'

    bag_img_dic = {}
    sides=['left','right']
    if bag_file_path.split('.')[-1] == 'bag':
        PKL = False
    elif bag_file_path.split('.')[-1] == 'pkl':
        PKL = True
    else:
        assert(False)

    if not PKL:
        bag_handler = Bagfile_Handler(bag_file_path,['/bair_car/zed/left/image_rect_color','/bair_car/zed/right/image_rect_color'])
        for side in sides:
            bag_img_dic[side] = {}
        
        topic, msg, timestamp = bag_handler.get_bag_content()
        while msg != None:            
            timestamp = round(timestamp.to_sec(),3)
            img = bridge.imgmsg_to_cv2(msg,color_mode)
            bag_img_dic[topic_name_map[topic]][timestamp] = img
            topic, msg, timestamp = bag_handler.get_bag_content()
    else:
        bag_img_dic = load_obj(bag_file_path)

    if include_flip:
        for side in sides:
            bag_img_dic[side+'_flip'] = {}
            for t in bag_img_dic[side]:
                img = bag_img_dic[side][t]
                bag_img_dic[side+'_flip'][t] = scipy.fliplr(img)
    return bag_img_dic


def save_images(bag_file_src_path,bag_file_dst_path):
    
    # If file already exists do nothing
    if os.path.isfile(os.path.join(bag_file_dst_path)):
        return
    
    bag_img_dic = load_images(bag_file_src_path,color_mode="rgb8",include_flip=False)
    for side in bag_img_dic:
        for t in bag_img_dic[side]:
            img = bag_img_dic[side][t]
            bag_img_dic[side][t] = cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
    print "Bag_File.load_images:: saving " + os.path.join(bag_file_dst_path)
    save_obj(bag_img_dic,os.path.join(bag_file_dst_path))


def bag_folder_save_images(bag_folder_src_path,bag_folder_dst_path):
    unix('mkdir -p '+bag_folder_dst_path)
    bag_file_paths = sorted(glob.glob(os.path.join(bag_folder_src_path,'*.bag')))
    for bf in bag_file_paths:
        bag_file_dst_path = os.path.join(bag_folder_dst_path)
        bag_file_dst_path = os.path.join(bag_file_dst_path,fname(bf)+'.pkl')
        print bag_file_dst_path
        save_images(bf,bag_file_dst_path)


def bag_folders_save_images(bag_folders_src_path,bag_folders_dst_path):
    bag_folders_paths = sorted(glob.glob(os.path.join(bag_folders_src_path,'*')))
    ef = sorted(os.path.join(bag_folders_dst_path,'*'),key=natural_keys)
    existing_folders = []
    for e in ef:
        existing_folders.append(fname(e))
    for bfp in bag_folders_paths:
        if fname(bfp) not in existing_folders:
            bag_folder_save_images(bfp,os.path.join(bag_folders_dst_path,fname(bfp)))
        else:
            cprint('Excluding '+bfp,'green','on_blue')


def bag_folders_transfer_meta(bag_folders_src_path,bag_folders_dst_path):
    bag_folders_paths = sorted([os.path.join(bag_folders_src_path,file) for file in os.listdir(bag_folders_src_path)],key=natural_keys)
    for bag_folder_path in bag_folders_paths:
        unix('mkdir -p '+os.path.join(bag_folders_dst_path,fname(bag_folder_path)))
        meta_dirs = sorted(glob.glob(os.path.join(bag_folder_path,'.pre*')))
        for meta_dir in meta_dirs:
            data = sorted(glob.glob(os.path.join(meta_dir,'left*')))
            data += sorted(glob.glob(os.path.join(meta_dir,'pre*')))
            for d in data:
                cprint(os.path.join(os.path.join(bag_folders_dst_path,fname(bag_folder_path))),'yellow')
                unix_str = d2s('scp ',d,os.path.join(bag_folders_dst_path,fname(bag_folder_path)))
                if len(glob.glob(os.path.join(bag_folders_dst_path,fname(bag_folder_path),fname(d)))) == 0: # test this first
                    cprint(unix_str,'red')
                    unix(unix_str)
                else:
                    cprint(d2s('NOT',unix_str))