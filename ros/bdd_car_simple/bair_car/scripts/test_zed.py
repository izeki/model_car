import rospy

from bair_car.ros_utils import ImageROSListener

import matplotlib.pyplot as plt

if __name__ == '__main__':
    rospy.init_node('test_zed', anonymous=True)
    camera = ImageROSListener('/bair_car/zed/right/image_rect_color')

    rospy.sleep(0.1)
    im = camera.get_image()

    plt.figure()
    plt.imshow(im)
    plt.show()

