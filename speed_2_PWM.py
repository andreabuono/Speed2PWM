#!/usr/bin/python

import rosbag
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import easygui
import numpy.polynomial.polynomial as poly
from array import array



bags=1

for i in range(bags):
	filename = easygui.fileopenbox()
	bag = rosbag.Bag(filename)
	M1_speed = []
	M2_speed = []
	M3_speed = []
	M4_speed = []
	
	M1_pwm =[]
	M2_pwm =[]
	M3_pwm =[]
	M4_pwm =[]

	for topic, msg, t in bag.read_messages(topics='/crazyflie1/m1'):
		if msg.values[1] >= 25000:
			M1_speed.append((168000000/msg.values[0])* 2 * np.pi)
			M1_pwm.append(msg.values[1]/65535)

	for topic, msg, t in bag.read_messages(topics='/crazyflie1/m2'):
		if msg.values[1] >= 25000:
			M2_speed.append((168000000/msg.values[0])* 2 * np.pi)
			M2_pwm.append(msg.values[1]/65535)

	for topic, msg, t in bag.read_messages(topics='/crazyflie1/m3'):
		if msg.values[1] >= 25000:
			M3_speed.append((168000000/msg.values[0])* 2 * np.pi)
			M3_pwm.append(msg.values[1]/65535)

	for topic, msg, t in bag.read_messages(topics='/crazyflie1/m4'):
		if msg.values[1] >= 25000:
			M4_speed.append((168000000/msg.values[0])* 2 * np.pi)
			M4_pwm.append(msg.values[1]/65535)
		
	bag.close()

M1_speed = np.array(M1_speed)
M1_pwm = np.array(M1_pwm)

M2_speed = np.array(M2_speed)
M2_pwm = np.array(M2_pwm)

M3_speed = np.array(M3_speed)
M3_pwm = np.array(M3_pwm)

M4_speed = np.array(M4_speed)
M4_pwm = np.array(M4_pwm)


plt.plot(M1_pwm, M1_speed, ".")
plt.plot(M2_pwm, M2_speed, ".")
plt.plot(M3_pwm, M3_speed, ".")
plt.plot(M4_pwm, M4_speed, ".")

coefs1 = poly.polyfit(M1_pwm, M1_speed, 1)
coefs2 = poly.polyfit(M2_pwm, M2_speed, 1)
coefs3 = poly.polyfit(M3_pwm, M3_speed, 1)
coefs4 = poly.polyfit(M4_pwm, M4_speed, 1)

b_coefs = [coefs1[0], coefs2[0], coefs2[0], coefs2[0]]
b_coef = np.mean(b_coefs)

M1_speed -= b_coef
M2_speed -= b_coef
M3_speed -= b_coef
M4_speed -= b_coef

x1 = M1_pwm[:,np.newaxis]
x2 = M2_pwm[:,np.newaxis]
x3 = M3_pwm[:,np.newaxis]
x4 = M4_pwm[:,np.newaxis]

m1_coef, _, _, _ = np.linalg.lstsq(x1, M1_speed)
m2_coef, _, _, _ = np.linalg.lstsq(x2, M2_speed)
m3_coef, _, _, _ = np.linalg.lstsq(x3, M3_speed)
m4_coef, _, _, _ = np.linalg.lstsq(x4, M4_speed)

print "**********************************************************************"
print "m Coeficients"
print m1_coef
print m2_coef
print m3_coef
print m4_coef

print "**********************************************************************"
print "b Coeficient"
print b_coef

coefs1 = [b_coef, m1_coef]
coefs2 = [b_coef, m2_coef]
coefs3 = [b_coef, m3_coef]
coefs4 = [b_coef, m4_coef]

x_new = np.linspace(0,1)

speed1 = poly.polyval(x_new, coefs1)
speed2 = poly.polyval(x_new, coefs2)
speed3 = poly.polyval(x_new, coefs3)
speed4 = poly.polyval(x_new, coefs4)

plt.plot(x_new, speed1)
plt.plot(x_new, speed2)
plt.plot(x_new, speed3)
plt.plot(x_new, speed4)


plt.show()

# m1_coefs = []
# m2_coefs = []
# m3_coefs = []
# m4_coefs = []

# for i in range(len(M1_speed)):
# 	m1_coefs.append((M1_speed[i]-b_coef)/M1_pwm[i])
# for i in range(len(M2_speed)):
# 	m2_coefs.append((M2_speed[i]-b_coef)/M2_pwm[i])
# for i in range(len(M3_speed)):
# 	m3_coefs.append((M3_speed[i]-b_coef)/M3_pwm[i])
# for i in range(len(M4_speed)):
# 	m4_coefs.append((M4_speed[i]-b_coef)/M4_pwm[i])

# m1_coef = np.mean(m1_coefs)
# m2_coef = np.mean(m2_coefs)
# m3_coef = np.mean(m3_coefs)
# m4_coef = np.mean(m4_coefs)

# print m1_coef
# print m2_coef
# print m3_coef
# print m4_coef

# coefs1 = [b_coef, m1_coef]
# coefs2 = [b_coef, m2_coef]
# coefs3 = [b_coef, m3_coef]
# coefs4 = [b_coef, m4_coef]

# x_new = np.linspace(0,1)

# speed1 = poly.polyval(x_new, coefs1)
# speed2 = poly.polyval(x_new, coefs2)
# speed3 = poly.polyval(x_new, coefs3)
# speed4 = poly.polyval(x_new, coefs4)

# plt.plot(x_new, speed1)
# plt.plot(x_new, speed2)
# plt.plot(x_new, speed3)
# plt.plot(x_new, speed4)


# plt.show()
