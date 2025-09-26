# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:24:54 2023

@author: eduardo.serralta
"""

#Diffraction calibration measurement










#processessing


import math

h = 1
k = 1
l = 1



measured_2_bragg_angle= 26.34/2 #milliradians # note that input is 2*theta_bragg



wl = 0.003701 #relativistic wavelength of electrons @ 100 keV


d_hkl_Si = math.sqrt(((0.543**2)/(h**2 + k**2 + l**2))) # nm #only valid for silicon
d_hkl_STO = math.sqrt(((0.3945**2)/(h**2 + k**2 + l**2)))


theoretical_2_bragg_angle_hkl_Si = 1000*((wl/d_hkl_Si)) #miliradians
theoretical_2_bragg_angle_hkl_STO = 1000*((wl/d_hkl_STO)) #miliradians


Error_Si= 100*((theoretical_2_bragg_angle_hkl_Si - measured_2_bragg_angle)/theoretical_2_bragg_angle_hkl_Si)
Error_STO= 100*((theoretical_2_bragg_angle_hkl_STO - measured_2_bragg_angle)/theoretical_2_bragg_angle_hkl_STO) 


experimental_Si_d_hkl = 1000*(wl/(measured_2_bragg_angle))
experimental_STO_d_hkl = 1000*(wl/(measured_2_bragg_angle))

d_error_si = 100*((experimental_Si_d_hkl- d_hkl_Si)/d_hkl_Si)
d_error_sto = 100*((experimental_STO_d_hkl- d_hkl_STO)/d_hkl_STO)


print("h=",h, "k= ", k, "l=", l)
print("theoretical_2_bragg_angle_hkl= ", theoretical_2_bragg_angle_hkl_Si)
print( "measured_2_bragg_angle =", measured_2_bragg_angle)
print("d_hkl_Si=", d_hkl_Si)
print( "Error_Si =" , Error_Si,"%")
print("d_error_si = ", d_error_si )
print("$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$")
print("d_hkl_STO=",d_hkl_STO)
print("theoretical_2_bragg_angle_hkl_STO = ", theoretical_2_bragg_angle_hkl_STO)
print("Error_STO= " , Error_STO, "%")
print("d_error_sto = ", d_error_sto, "%")