import rospy

from bair_car.car import Car

if __name__ == '__main__':
    rospy.init_node('test_car', anonymous=True)
    car = Car()
    
    import IPython; IPython.embed()
    
