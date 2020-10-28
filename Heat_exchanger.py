import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class Heat_ex():
	
	def __init__(self, T_hot_in, T_cold_in, Current, Cold_feed_rate, Hot_feed_rate):
		#Co-current : True, Counter-current : False
		self.T_hot_in = T_hot_in
		self.T_cold_in = T_cold_in
		self.Cold_feed_rate = Cold_feed_rate*1.667*10**(-5)		#m^3/s
		self.Hot_feed_rate = Hot_feed_rate*1.667*10**(-5)			#m^3/s
		
		self.flow_type = Current
		
		self.l = 1500*10**(-3) 									#total length of heat exchanger in m
		self.ID_in = 15*10**(-3)								#inner tube ID in m
		self.OD_in = 15.88*10**(-3)							#inner tube OD in m
		self.ID_out = 40*10**(-3)								#outer tube ID in m
		self.DH_in = self.ID_in
		self.DH_out = self.ID_out
		
		self.k_t = 20														#thermal conductivity of tube
		self.Cold_v = self.Cold_feed_rate/((np.pi/4)*(self.ID_out**2-self.OD_in**2))		#m/s
		self.Hot_v = self.Hot_feed_rate/((np.pi*self.ID_in**2)/4)		#m/s
		
		
		T = sp.Symbol('T')
		self.mu_w = 2.414*10**(-5) * 10**(247.8/(T-140))		#[1], T in K, viscosity in Pa*s
		self.rho_w = (999.83952 + 16.945176*(T-273) - 7.9870401*10**(-3)*(T-273)**2 - 46.170461*10**(-6)*(T-273)**3 + 105.56302*10**(-9)*(T-273)**4 - 280.54253*10**(-12)*(T-273)**5)/(1+16.897850*10**(-3)*(T-273)) #[2], T in K, density in kg/m^3
		self.k_w = -8.354*10**(-6)*T**2 + 6.53*10**(-3)*T - 0.5981 #[3], T in K, thermal conductivity in W/(m*K)
		self.Cp_w = (-203.606 + 1523.290*(T/1000) - 3196.413*(T/1000)**2 + 2474.455*(T/1000)**3 +3.855326*(T/1000)**(-2))*1000/18 #[4], T in K, heat capacity in J/(kg*K)
		
	def h_calc(self, u, D, temp):
		#Pr = Cp mu / k
		#Re = rho*u*D/mu
		#Nu = 0.012(Re^0.87 - 280)Pr^0.4
		#Nu = h DH / k
		
		T = sp.Symbol('T')
		rho = float(self.rho_w.subs(T, temp))
		mu = float(self.mu_w.subs(T, temp))
		Cp = float(self.Cp_w.subs(T, temp))
		k = float(self.k_w.subs(T, temp))
		
		Pr = Cp*mu/k
		Re = rho*u*D/mu
		Nu = 0.012*((Re)**0.87 - 280)*Pr**0.4

		return Nu*k/D
	
	def mass_flow_rate(self, Q, temp):
		#return kg/s
		T = sp.Symbol('T')
		result = Q*self.rho_w.subs(T, temp)
		return result
	
	def main_calc(self):
		#h can be calculated by self.h(u, D)
		#mass flow rate can be calculated by self.mass_flow_rate(Q)
		#hot flow -> inner tube, cold flow -> outer tube
		
		T = sp.Symbol('T')
		
		dx = self.l/3000
		x_array = np.arange(0, self.l+dx, dx)
		
		A_i = np.pi*self.ID_in*dx
		A_o = np.pi*self.ID_out*dx
		
		r_o = self.ID_out/2
		r_i = self.ID_in/2
		k_t = self.k_t
		l = dx
		
		
		T_in = np.zeros_like(x_array)
		T_out = np.zeros_like(x_array)
		T_in[0] = self.T_hot_in
		T_out[0] = self.T_cold_in
		
		q_tot=0
		
		for i in range(len(x_array)-1):
			#Cp J/(kg*K)
			#m kg/s
			#rho kg/m^3
			#q J/s
			h_i = self.h_calc(self.Hot_v, self.DH_in, T_in[i])
			h_o = self.h_calc(self.Cold_v, self.DH_out, T_out[i])

			m_c = self.mass_flow_rate(self.Cold_feed_rate, T_out[i])
			m_h = self.mass_flow_rate(self.Hot_feed_rate, T_in[i])
			
			UA = 1/(1/(h_i*A_i)+1/(h_o*A_o)+sp.log(r_o/r_i)/(2*np.pi*k_t*l))
			
			q = UA*(T_in[i]-T_out[i])
			q_tot += q
			delT_in = q/(self.Cp_w.subs(T, T_in[i])*m_h)
			delT_out = q/(self.Cp_w.subs(T, T_out[i])*m_c)
			T_in[i+1] = T_in[i] - delT_in
			T_out[i+1] = T_out[i] + delT_out
			
		plt.close()
		plt.plot(x_array, T_in, label = 'Hot water')
		plt.plot(x_array, T_out, label = 'Cool water')
		plt.grid()
		plt.legend()
		plt.xlabel('x in (m)')
		plt.ylabel('T in (K)')
		plt.title('Fluid temp vs passed lenght')
		plt.show()
		
		print('Hot_end = {}'.format(T_in[-1]-273))
		print('Cool_end = {}'.format(T_out[-1]-273))
		print('q = {}'.format(q_tot))

if __name__ == '__main__':

	
	test1 = Heat_ex(40+273, 19+273, True, 4, 4)
	test1.main_calc()
	print("Co current 3:3 40 vs 19 'C")
	
	
	test2 = Heat_ex(40+273, 19+273, True, 8, 4)
	test2.main_calc()
	print("Co current 3:6 40 vs 19 'C")
	
	
	test3 = Heat_ex(60+273, 19+273, True, 4, 4)
	test3.main_calc()
	print("Co current 3:3 60 vs 19 'C")
	
	
	test4 = Heat_ex(61+273, 19+273, True, 8, 4)
	test4.main_calc()
	print("Co current 3:6 60 vs 19 'C")
	
"""
Reference
[1] : https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
[2] : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/
[3] : W. Kays, M. Crawford, B. Weigand, Convective Heat and Mass Transfer, fourth ed., McGraw-Hill, Singapore, 2005.
[4] : https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=JANAFL&Plot=on
"""
