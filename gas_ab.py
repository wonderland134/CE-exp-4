'''
Empirically, k_L*a and k_G*a can be expressed c1 * G_x^c2 and c3 * G_y^c4 * G_x^c5 respectively.
So this file has purpose that find the c1, c2, c3, c4, c5 value to fit our data.
This value can be used gas absorption column exist in Ajou university for chemical engineering experiment 4 gas absorption extrument because it will be affected by type of packing material.
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class gas_ab():
	def __init__(self):
		
		self.Column_ID = 70*10**(-3)			# in m
		self.Column_l = 1.4							# in m
		
		self.T_init = 13.8+273			#T in K
		self.T_last = 14.5+273			#T in K
		self.T_av = (self.T_init+self.T_last)/2
		
		self.H = 150.33-8498.72/self.T_av-20.0841*sp.log(self.T_av)+7.4*10**(-4)		#T in K, H in atm, henry constant
		self.rho_w = (999.83952 + 16.945176*(self.T_av-273) - 7.9870401*10**(-3)*(self.T_av-273)**2 - 46.170461*10**(-6)*(self.T_av-273)**3 + 105.56302*10**(-9)*(self.T_av-273)**4 - 280.54253*10**(-12)*(self.T_av-273)**5)/(1+16.897850*10**(-3)*(self.T_av-273)) #[2], T in K, density in g/L
		
		
		self.x_top = 1.13*10**(-5)
		self.x_bot = 2.034*10**(-5)
		
		self.Water_flow_rate = 2*self.rho_w/18										#mol/min
		self.Air_flow_rate = 10/(0.0821*self.T_av)								#mol/min
		self.CO2_flow_rate = 5/(0.0821*self.T_av)									#mol/min
		
		self.V = self.Air_flow_rate + self.CO2_flow_rate					#mol/min
		self.L = self.Water_flow_rate															#mol/min
		
		self.y_bot = self.CO2_flow_rate/(self.Air_flow_rate+self.CO2_flow_rate)
		self.y_top = self.y_bot-(self.x_bot-self.x_top)*self.L/self.V

	def calc(self):
		#x_top and y_bot can be used but other information(x_bot, y_top) must do not be used.
		mu = 2.414*10**(-5) * 10**(247.8/(self.T_av-140))*1000		#T in K, viscosity in cP
		L = 2*self.rho_w/(np.pi*self.Column_ID**2/4)*0.01229 # lb/(h*ft^2)
		F_min = 10**(-6)
		F_max = 10**(-4)
		n = 50
		dF = (F_max - F_min)/n
		F = np.arange(F_min, F_max+dF, dF)
		
		KGa = F*(L/mu)**(2/3)*267 #mol/(m^3 min atm)
		'''
		L : liquid-flow rate in lb/(h*ft^2)
		mu : viscosity, centipoises
		'''
		Gm = self.V/(np.pi*self.Column_ID**2/4)
		y_sat = self.x_top*self.H/1
		y_init = self.y_bot
		y = sp.Symbol('y')
		y_array =np.zeros_like(KGa)
		for i in range(len(KGa)):
			eq = Gm/KGa[i]*(sp.log(abs(1-y))-sp.log(abs(y-y_sat)))/(1-y_sat)
			
			y_temp = y_init
			while True:
				det = eq.subs(y,y_temp)-eq.subs(y,y_init)
				if det > self.Column_l:
					y_array[i]=y_temp
					if y_temp<self.y_top:
						target=i
					break
				else:
					y_temp = y_temp-0.00001
		
		plt.close()
		plt.plot(F, y_array)
		plt.plot([F[0], F[-1]], [self.y_top, self.y_top])
		plt.grid()
		plt.title('F vs expected y_top')
		plt.xlabel('F')
		plt.ylabel('y_top')
		plt.show()
		print('target F = {}'.format(F[target]))
	
	def print_all(self):
		print('y_top : {}'.format(self.y_bot))
		print('y_bot : {}'.format(self.y_top))
		
if __name__ == '__main__':
	test = gas_ab()
	test.print_all()
	test.calc()
