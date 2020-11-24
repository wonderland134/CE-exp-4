import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class calc_data():
	def __init__(self, t_data, h_data, T, P_data):
		
		#############Experiment data############
		self.t_data = t_data		#s
		self.h_data = h_data		#cm
		self.P_data = P_data		#atm
		self.T = np.round_(T,1)	#K
		self.P_av = np.round_(np.average(P_data),3)		#atm
		#############Experiment data############
		
		############Substance properties############
		self.M_acetone = 58.08 #g/mol
		self.M_air = 28.851 #g/mol
		self.sigma_acetone = 4.6			#A
		self.sigma_air = 3.711				#A
		self.sigma = (self.sigma_acetone+self.sigma_air)/2
		self.epsilon_acetone = 560.2	#K, note that epsilon/k
		self.epsilon_air = 78.6				#K, note that epsilon/k
		self.epsilon = (self.epsilon_acetone*self.epsilon_air)**0.5
		self.collision_integral = self.calc_collision_integral(self.T)
		self.P_set = 10**(4.42448 -(1312.253/(self.T-32.445)))*0.9869 #atm
		#reference by (1) 
		
		self.Tb_acetone =	329.4						#nomal B.P in K
		self.Tb_air = 80									#nomal B.P in K
		self.Tc_acetone =	508.2						#critlcal T in K
		self.Tc_air =	132.2								#critlcal T in K
		self.Pc_acetone = 47.01 * 0.9869	#critlcal P in atm
		self.Pc_air =	37.45 * 0.9869			#critlcal P in atm
		self.Vc_acetone = 209							#critical molar volume in cm^3/mol
		self.Vc_air =	84.8								#critical molar volume in cm^3/mol 
		self.Vb_acetone, self.Vb_air = self.liquid_density()	#cm^3/mol at normal BP
		#reference by (2), (3)
		#reference by (8) molar volume of air is 29.9 cm^3/mol
		self.Vb_air=29.9
		
		self.Vd_acetone = 16.5*3 + 1.98*6 + 5.48*1		#diffusion volume
		self.Vd_air = 20.1
		#reference by (7)
		
		self.Acetone_dens = 57.6214/(0.233955**(1+(1-self.T/507.803)**0.254167))/1000 #g/cm^3
		#reference by (4)
		
		self.Acetone_dipole = 2.88
		#reference by (9)
		############Substance properties############
		
		#############Result############
		self.D_exp = None
		self.D_exp_new = None
		self.D_chap = None
		self.D_brokaw = None
		self.D_chen = None
		self.D_fuller = None
		self.D_ref = None
		self.delta_h_1 = None
		self.delta_h_2 = None
		#############Result############
	
	def liquid_density(self):
		#reference by (5)
		Tb_N = 77.3
		Tb_O = 90.2
		Tb_acetone = self.Tb_acetone
		
		C1_N, C2_N, C3_N, C4_N = (3.2091, 0.2861, 126.2, 0.2966)
		d_N = C1_N/C2_N**(1+(1-Tb_N/C3_N)**C4_N) * 0.001		#d in mol/cm^3, T in K
		
		C1_O, C2_O, C3_O, C4_O = (3.9143, 0.28772, 154.58, 0.2924)
		d_O = C1_O/C2_O**(1+(1-Tb_O/C3_N)**C4_N) * 0.001
		
		d_air = d_N*0.79 + d_O*0.21
		
		C1_acetone, C2_acetone, C3_acetone, C4_acetone = (1.2332, 0.25886, 508.2, 0.2913)
		d_acetone = C1_acetone/C2_acetone**(1+(1-Tb_acetone/C3_acetone)**C4_acetone) * 0.001
		
		return (1/d_acetone, 1/d_air)
		
	def calc_collision_integral(self, T):
		Dimensionless_T = T/self.epsilon

		if Dimensionless_T < 0.3 or Dimensionless_T > 400:
			print('dimensionless temperature is out of range')
			return 0
		
		DT_data = np.load('Dimensionless_T.npy')
		C_data = np.load('Collision_integral.npy')
		for i in range(0,len(DT_data)):
			det = Dimensionless_T - DT_data[i]
			if det<0:
				result = (C_data[i]-C_data[i-1])/(DT_data[i]-DT_data[i-1])*(Dimensionless_T-DT_data[i])+C_data[i]
				return result
	
	def poly_regression(self, x, y, n, title = '', Figure = True):
		plt.close()
		plt.plot(x, y, 'bo', label = 'Data')
		
		result = np.polyfit(x, y, n)
		if Figure == True:
			if n == 1:
				fit_x = [x[0],x[len(x)-1]]
				fit_y = [result[0]*fit_x[0]+result[1], result[0]*fit_x[1]+result[1]]
		
				if result[1]<0:
					plt.plot(fit_x, fit_y, 'r', label = 'y={}x{}'.format(np.round_(result[0], 4), np.round_(result[1], 4)))
				else:
					plt.plot(fit_x, fit_y, 'r', label = 'y={}x+{}'.format(np.round_(result[0], 4), np.round_(result[1], 4)))
			elif n == 2:
				x_fit = np.arange(x[0], x[len(x)-1]+1, 1)
				x = sp.Symbol('x')
				y_fit = result[0]*x_fit**2 + result[1]*x_fit + result[2]
			
				plt.plot(x_fit, y_fit, 'r', label = 'y={}x^2 + {}x + {}'.format(result[0], result[1], np.round_(result[2], 4)))
			
			plt.grid()
			plt.legend()
			plt.title(title)
			plt.xlabel('x axis')
			plt.ylabel('y axis')
			plt.show()
		
		return result
		
	def reference_D(self):
		#0'C 1atm -> D = 0.109 cm^2/s
		#reference by (6)
		P = self.P_av
		T = self.T
		D = 0.109 * (1/P)*(T/273.15)**1.75
		self.D_ref = np.round_(D, 4)
	
	def exp_D(self):
		P_1 = self.P_set
		P_2 = 0
		P = self.P_av
		R = 82.06				# cm^3 atm / (mol K)
		rho = self.Acetone_dens		#g/cm^3
		T = self.T
		M_A = self.M_acetone

		x_data = self.t_data
		y_data = self.h_data**2-self.h_data[0]**2
		slope = self.poly_regression(x_data, y_data, 1, 't vs hf^2 - hi^2')[0]
		
		Plm = (P-P_2-(P-P_1))/(np.log(P-P_2)-np.log(P-P_1))
		D = slope*rho*R*T*Plm/(2*M_A*P*P_1)
		self.D_exp = np.round_(D, 4)
	
	def exp_D_new(self):	
		x_data = self.t_data
		y_data = self.P_data
		
		P_1 = self.P_set
		M_A = self.M_acetone
		T = self.T
		rho = self.Acetone_dens
		R = 82.06			# cm^3 atm / (mol K)
		
		a, b, c = self.poly_regression(x_data, y_data, 2, 't vs P', Figure = False)
		
		t = sp.Symbol('t')
		
		P = a*t**2 + b*t + c
		pi = P/(P_1/(sp.log(P/(P-P_1))))
		
		X_data = np.array([])
		delt = 1
		ds = 0
		t_cord = 0
		for i in range(len(x_data)):
			while True:
				if t_cord != x_data[i]:
					t_cord += delt
					ds += float(pi.subs(t, t_cord)*delt)
				else:
					X_data = np.hstack(( X_data, np.round(np.array([ds]), 4) ))
					break
		
		Y_data = self.h_data**2-self.h_data[0]**2

		slope = self.poly_regression(X_data, Y_data, 1, 'pi(t)-pi(0) vs hf^2-hi^2', Figure = False)[0]
		D = slope/2*R*rho*T/(M_A*P_1)
		
		self.D_exp_new = np.round_(D, 4)
		
		
	def chap_D(self):
		T = self.T
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		sigma = self.sigma
		C_I = self.collision_integral
		
		def func(T, MA, MB, P, sigma, omega):
			return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
		D = func(T, M_A, M_B, P, sigma, C_I)
		
		self.D_chap = np.round_(D, 4)
		
	def brokaw_D(self):
		delta = 0 # Because dipole moment of air = 0, delta of air = 0
		Tb_A = self.Tb_acetone
		Tb_B = self.Tb_air
		Vb_A = self.Vb_acetone
		Vb_B = self.Vb_air
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		T = self.T
		mu_A = self.Acetone_dipole
		
		#Calculate delta value
		def func_delta(Vb, Tb, mu):
			return 1.94*10**3*mu**2/(Vb*Tb)
			
		delta_acetone = func_delta(Vb_A, Tb_A, mu_A)
		delta_air = 0
		
		#Calculate epsilon value
		def func_epsilon(delta, Tb):
			return 1.18*(1+1.3*delta**2)*Tb
		
		epsilon_acetone = func_epsilon(delta_acetone, Tb_A)
		epsilon_air = func_epsilon(delta_air, Tb_B)
		epsilon = (epsilon_air * epsilon_acetone)**0.5
		
		D_T = T/epsilon			#dimensionless temperature
		
		#Calculate collision integral by dimensionless temperature
		def func_col_int(delta, D_T):
			A, B, C, D, E, F, G, H = (1.06036, 0.15610, 0.19300, 0.47635, 1.03587, 1.52996, 1.76474, 3.89411)
			return A/D_T**B + C/np.exp(D*D_T)+E/np.exp(F*D_T)+G/np.exp(H*D_T)+0.196*delta**2/D_T
		
		collision_integral = func_col_int(delta, D_T)
		
		#Calculate sigma value
		def func_sigma(delta, Vb):
			return (1.585 * Vb/(1+1.3*delta**2))**(1/3)
		
		sigma_acetone = func_sigma(delta, Vb_A)
		sigma_air = func_sigma(delta, Vb_B)
		sigma = (sigma_acetone*sigma_air)**0.5		
		
		#Calculate diffusivity by brokaw corrected factors
		def func(T, MA, MB, P, sigma, omega):
			return (0.001858*T**1.5*(1/MA+1/MB)**0.5)/(P*sigma**2*omega)
		
		D = func(T, M_A, M_B, P, sigma, collision_integral)
		self.D_brokaw = np.round_(D, 4)
		
	def chen_D(self):
		M_A = self.M_acetone
		M_B = self.M_air
		P = self.P_av
		T = self.T
		Tc_A = self.Tc_acetone
		Tc_B = self.Tc_air
		Vc_A = self.Vc_acetone
		Vc_B = self.Vc_air
		
		def func(T, MA, MB, P, TcA, TcB, VcA, VcB):
			return 0.01498*T**1.81*(1/MA+1/MB)**0.5/(P*(TcA*TcB)**0.1405*(VcA**0.4+VcB**0.4)**2)
		
		D = func(T, M_A, M_B, P, Tc_A, Tc_B, Vc_A, Vc_B)
		
		self.D_chen = np.round_(D, 4)
	
	def fuller_D(self):
		Vd_A = self.Vd_acetone
		Vd_B = self.Vd_air
		T = self.T
		P = self.P_av
		M_A = self.M_acetone
		M_B = self.M_air		
		
		def func(T, MA, MB, P, VA, VB):
			return 0.001*T**1.75*(1/MA+1/MB)**0.5/(P*(VA**(1/3)+VB**(1/3))**2)
			
		D = func(T, M_A, M_B, P, Vd_A, Vd_B)
		
		self.D_fuller = np.round_(D, 4)
		
	def find_h_1(self):
		#del_h is positive value -> actual length is larger than measured
		del_h_start = -3					#in cm
		del_h_end = 3
		del_h_del = 0.1
		del_h = np.arange(del_h_start, del_h_end + del_h_del, del_h_del)
		h_data = self.h_data
		
		P_1 = self.P_set
		P_2 = 0
		P = self.P_av
		R = 82.06				# cm^3 atm / (mol K)
		rho = self.Acetone_dens
		T = self.T
		M_A = self.M_acetone
		
		Plm = (P-P_2-(P-P_1))/(np.log(P-P_2)-np.log(P-P_1))
		
		D_array = np.array([])
		for i in range(len(del_h)):
			x_data = self.t_data
			y_data = (h_data + del_h[i])**2 - (h_data[0] + del_h[i])**2
			slope = self.poly_regression(x_data, y_data, 1, Figure = False)[0]
			temp = slope*rho*R*T*Plm/(2*M_A*P*P_1)
			
			if i>1:
				det = (self.D_ref-temp)*(self.D_ref-D_array[-1])
				if det<0:
					self.delta_h_1 = np.round_(del_h[i],3)
			
			D_array = np.hstack([D_array, np.array([temp])])
			
			
		plt.close()
		plt.plot(del_h, D_array)
		plt.plot([del_h_start, del_h_end], [self.D_ref, self.D_ref])
		plt.xlabel('Variation of h')
		plt.ylabel('Diffusivity in cm^2/s')
		plt.title('Diffusivity change If h-variation exist')
		plt.grid()
		plt.show()
		
	def find_h_2(self):
		del_h_start = -3					#in cm
		del_h_end = 3
		del_h_del = 0.1
		del_h = np.arange(del_h_start, del_h_end + del_h_del, del_h_del)
		h_data = self.h_data
		
		x_data = self.t_data
		intercept_data = np.zeros_like(del_h)
		for i in range(len(del_h)):
			y_data = (h_data + del_h[i])**2 - (h_data[0] + del_h[i])**2
			intercept_data[i] = self.poly_regression(x_data, y_data, 1, Figure = False)[1]
		
		plt.close()
		plt.plot(del_h, intercept_data)
		plt.xlabel('Variation of h')
		plt.ylabel('y-intercept of regression data')
		plt.title('intercept change If h-variation exist')
		plt.grid()
		plt.show()

				
	def calc_all(self):
		self.reference_D()
		self.exp_D()
		self.chap_D()
		self.brokaw_D()
		self.chen_D()
		self.fuller_D()
		self.exp_D_new()
	
	def print_result(self):
		print('*Result of calculation*')
		print('\tReference result : ' + str(self.D_ref) + ' cm^2/s')
		print('\tExperimental result : ' + str(self.D_exp) + ' cm^2/s')
		#print('\tExperimental result(new) : ' + str(self.D_exp_new) + ' cm^2/s')
		print('\tChapman-Enskog Equation result : ' + str(self.D_chap) + ' cm^2/s')
		print('\tBrokaw method : ' + str(self.D_brokaw) + 'cm^2/s')
		print('\tChen and othmer equation : ' + str(self.D_chen) + 'cm^2/s')
		print('\tFuller, Schettler and gidding equation : ' + str(self.D_fuller) + 'cm^2/s')
		
	def print_condition(self):
		print('*Expriment condition (T and P)*')
		print('\tTemperature(average) : {} K'.format(self.T))
		print('\tPressure(average) : {} atm'.format(self.P_av))
		
	def print_error(self):
		plt.close()
		x_list = np.array([1, 2, 3, 4, 5])
		name_tag = ['Exp', 'Chapman', 'Brokaw', 'Chen', 'Fuller']
		y_list = np.array([self.D_exp, self.D_chap, self.D_brokaw, self.D_chen, self.D_fuller])
		y_list = 100*(y_list - self.D_ref)/self.D_ref
		y_list = np.abs(y_list)
		
		plt.xticks(x_list, name_tag)
		plt.bar(x_list, y_list, align = 'center', width = 0.4)
		plt.xlabel('Type')
		plt.ylabel('Error(%)')
		plt.title('Diffusivity error base on D_ref')
		plt.grid()
		plt.show()
		
		y_list = np.round_(y_list, 2)
		
		print('Exp : {}%'.format(y_list[0]))
		print('Chapman : {}%'.format(y_list[1]))
		print('Brokaw : {}%'.format(y_list[2]))
		print('Chen : {}%'.format(y_list[3]))
		print('Fuller : {}%'.format(y_list[4]))
		
	def Unstst_analysis(self, D_guess):
		R=82.06
		MA = self.M_acetone
		T = self.T
		rho = 57.6214/(0.233955**(1+(1-T/507.803)**0.254167))/1000

		x0 = -self.h_data[0] 			#cm
		D = D_guess 						#cm^2/s
		P_s = self.P_set					#B.C where x = 0
		P_e = 0										#B.C where x = L
		P = self.P_av

		n_x = 50
		dx_0 = abs(x0)/n_x

		t_end = self.t_data[-1]			#s
		n_t = 2000000
		dt = t_end/n_t		#s
		t_array = np.linspace(0, t_end, n_t)

		x1_array = np.zeros_like(t_array)
		x1_array[0] = x0

		#P_array[t][x], P_array[i][j]

		#P[i+1][j]=dt/dx(dx/dt * P[i][j] + PD/(P[i][j-1]-P) * (P[i][j]-P[i][j-1])/dx -PD/(P[i][j]-P)(P[i][j+1]-P[i][j])/dx)
		progress = int(0)
		print(str(progress) + '% done')

		P_array = [np.zeros_like(np.linspace(0, abs(x0), n_x))]

		for i in range(len(t_array)):
			dx = np.abs(x1_array[i]/n_x)			#cm
			P_bef = P_array[i]
			P_bef[0] = P_s
			P_bef[-1] = P_e
		
			progress_temp = int(i/len(t_array)*100)
			if progress_temp != progress:
				progress = progress_temp
				print('{}'.format(progress) + '% done')
			
			while True:
				P_next = np.zeros_like(P_array[i])+(P_s+P_e)/2
				P_next[0] = P_s
				P_next[-1] = P_e
			
				for j in range(1, len(P_array[i])-1):
					first = P_bef[j]
					second = (P_bef[j+1]-2*P_bef[j]+P_bef[j-1])/(P-P_bef[j])
					third = ((P_bef[j]-P_bef[j-1])/(P-P_bef[j]))**2
					P_next[j] = first + dt*P*D/dx**2*(second+third) 
			
				det = (P_next - P_bef)/(P_bef+(P_s+P_e)/2)
			
				if abs(det.max()) < 0.0001:
					break
				else:
					P_bef = P_next
			if t_array[i] != t_array[-1]:
				x1_array[i+1] = x1_array[i]+dt*MA*P*D/(rho*R*T*(P-P_s))*(P_bef[1]-P_s)/dx 
			P_array.append(P_next)
	
		plt.close()
		print_percent = [0, 0.00005, 0.001, 0.005, 0.05, 0.3, 1]
		for i in print_percent:
			i = int(i * (len(t_array)-1))
			x_array = np.linspace(0, abs(x1_array[i]), n_x)
			plt.plot(x_array, P_array[i], label='t = {} s, L = {}'.format(np.round_(t_array[i], 2), np.round_(abs(x1_array[i]), 2)))
	
		plt.xlabel('x-axis (cm)')
		plt.ylabel('Partial pressure of A(atm)')
		plt.title('Unsteady state diffusion')
		plt.legend(loc='upper right')
		plt.grid()
		plt.show()
	
		plt.close()
		plt.plot(t_array, np.abs(x1_array))
		plt.plot(self.t_data, self.h_data, 'ro')
		plt.grid()
		plt.xlabel('time(s)')
		plt.ylabel('diffusion length(cm)')
		plt.title('t vs diffusioon length')
		plt.show()
		
	def Unstst_D(self, D_guess):
		
		R=82.06
		MA = self.M_acetone
		T = self.T
		rho = 57.6214/(0.233955**(1+(1-T/507.803)**0.254167))/1000

		x0 = -self.h_data[0] 			#cm
		D = D_guess 						#cm^2/s
		P_s = self.P_set					#B.C where x = 0
		P_e = 0										#B.C where x = L
		P = self.P_av

		n_x = 50
		dx_0 = abs(x0)/n_x

		t_end = self.t_data[-1]			#s
		n_t = 2000000
		dt = t_end/n_t		#s
		t_array = np.linspace(0, t_end, n_t)

		x1_array = np.zeros_like(t_array)
		x1_array[0] = x0

		#P_array[t][x], P_array[i][j]

		#P[i+1][j]=dt/dx(dx/dt * P[i][j] + PD/(P[i][j-1]-P) * (P[i][j]-P[i][j-1])/dx -PD/(P[i][j]-P)(P[i][j+1]-P[i][j])/dx)
		progress = int(0)
		print(str(progress) + '% done')

		P_array = [np.zeros_like(np.linspace(0, abs(x0), n_x))]

		for i in range(len(t_array)):
			dx = np.abs(x1_array[i]/n_x)			#cm
			P_bef = P_array[i]
			P_bef[0] = P_s
			P_bef[-1] = P_e
		
			progress_temp = int(i/len(t_array)*100)
			if progress_temp != progress:
				progress = progress_temp
				print('{}'.format(progress) + '% done')
			
			while True:
				P_next = np.zeros_like(P_array[i])+(P_s+P_e)/2
				P_next[0] = P_s
				P_next[-1] = P_e
			
				for j in range(1, len(P_array[i])-1):
					first = P_bef[j]
					second = (P_bef[j+1]-2*P_bef[j]+P_bef[j-1])/(P-P_bef[j])
					third = ((P_bef[j]-P_bef[j-1])/(P-P_bef[j]))**2
					P_next[j] = first + dt*P*D/dx**2*(second+third) 
			
				det = (P_next - P_bef)/(P_bef+(P_s+P_e)/2)
			
				if abs(det.max()) < 0.0001:
					break
				else:
					P_bef = P_next
			if t_array[i] != t_array[-1]:
				x1_array[i+1] = x1_array[i]+dt*MA*P*D/(rho*R*T*(P-P_s))*(P_bef[1]-P_s)/dx 
			P_array.append(P_next)
		
		def interpolation(target, x_array, y_array):
			for i in range(len(x_array)-1):
				if (target-x_array[i+1])*(target-x_array[i])<=0:
					return (y_array[i+1]/y_array[i])*(target-x_array[i])+y_array[i]
		
		temp_h = []
		for i in range(len(self.t_data)):
			temp_h.append(interpolation(t_data[i], t_array, -x1_array))
		
		temp_h = np.array(temp_h)
		dev= np.sum((self.h_data-temp_h)**2)
		
		return dev
	
	def find_unstst_D(self,D_guess , diff, n):
		D_array = np.linspace(D_guess-diff, D_guess+diff, n)
		dev_array = np.zeros_like(D_array)
		
		for i in range(len(D_array)):
			print('{} th process / {} process'.format(i, len(D_array)))
			dev_array[i] = self.Unstst_D(D_array[i])
		
		index = np.argmin(dev_array)
		
		self.Unstst_analysis(D_array[index])
		print(D_array[index])
		
		
		
if __name__ == '__main__':
	
	#1
	print('1st team')
	t_data = np.arange(0,3900,300)	#s
	h_data = np.array([5.28, 5.3, 5.31, 5.31, 5.39, 5.42, 5.43, 5.45, 5.48, 5.51, 5.54, 5.59, 5.64]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129, 130, 130, 128, 130, 129, 128])*9.678411*10**(-5)	#atm
	team_1st = calc_data(t_data, h_data, 46+273.15, 1-P_data)
	team_1st.calc_all()
	#team_1st.find_h_1()
	#team_1st.find_h_2()
	#team_1st.find_unstst_D(team_1st.D_ref, 0.02, 10)
	

	#2
	print('2nd team')
	t_data = np.arange(0,3900,300)	#s
	h_data = np.array([5.24, 5.3, 5.31, 5.35, 5.39, 5.42, 5.44, 5.47, 5.48, 5.53, 5.56, 5.59, 5.64]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129, 130, 130, 128, 130, 129, 128])*9.678411*10**(-5)	#atm
	team_2nd = calc_data(t_data, h_data, 45+273.15, 1-P_data)
	team_2nd.calc_all()
	#team_2nd.find_h_1()
	#team_2nd.find_h_2()
	#team_2nd.find_unstst_D(team_2nd.D_ref, 0.02, 10)
	
	#3
	print('3rd team')
	t_data = np.arange(0,2700,300)	#s
	h_data = np.array([3.3, 3.3, 3.4, 3.41, 3.5, 3.52, 3.55, 3.6, 3.7]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129, 130, 130])*9.678411*10**(-5)	#atm
	T_data = np.array([45, 44.5, 44.6, 44.5, 44.2, 44.2, 45, 44.7, 44.5])+273.15
	team_3rd = calc_data(t_data, h_data, np.average(T_data), 1-P_data)
	team_3rd.calc_all()
	#team_3rd.find_h_1()
	#team_3rd.find_h_2()
	#team_3rd.find_unstst_D(team_3rd.D_ref, 0.02, 10)
	
	#4
	print('4th team')
	t_data = np.arange(0,2400,300)	#s
	h_data = np.array([3.5, 3.57, 3.59, 3.6, 3.62, 3.65, 3.7, 3.72]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129, 130])*9.678411*10**(-5)	#atm
	T_data = np.array([45, 45, 45, 45, 45, 45, 44.5, 44.5])+273.15
	team_4th = calc_data(t_data, h_data, np.average(T_data), 1-P_data)
	team_4th.calc_all()
	#team_4th.find_h_1()
	#team_4th.find_h_2()
	#team_4th.find_unstst_D(team_4th.D_ref, 0.02, 10)
	
	#5
	print('5th team')
	t_data = np.arange(0,2100,300)	#s
	h_data = np.array([5.3, 5.32, 5.33, 5.39, 5.46, 5.47, 5.51]) 	#cm
	P_data = np.array([121, 126, 127, 128, 128, 129, 129])*9.678411*10**(-5)	#atm
	T_data = np.array([46, 46, 46, 45.5, 45.1, 45, 45])+273.15
	team_5th = calc_data(t_data, h_data, np.average(T_data), 1-P_data)
	team_5th.calc_all()
	#team_5th.find_h_1()
	#team_5th.find_h_2()
	#team_5th.find_unstst_D(team_4th.D_ref, 0.02, 10)
	
	'''
	#overal result
	print('')
	print('-------------------------------------')
	print('---------------Result----------------')
	print('-------------------------------------')
	print('1st team : {}cm^2/s, T = {}K, P = {}atm'.format(team_1st.D_exp, team_1st.T, team_1st.P_av))
	print('2nd team : {}cm^2/s, T = {}K, P = {}atm'.format(team_2nd.D_exp, team_2nd.T, team_2nd.P_av))
	print('3rd team : {}cm^2/s, T = {}K, P = {}atm'.format(team_3rd.D_exp, team_3rd.T, team_3rd.P_av))
	print('4th team : {}cm^2/s, T = {}K, P = {}atm'.format(team_4th.D_exp, team_4th.T, team_4th.P_av))
	print('5th team : {}cm^2/s, T = {}K, P = {}atm'.format(team_5th.D_exp, team_5th.T, team_5th.P_av))
	
	print('')
	print('h deviation for each teams, delta_h is positive value -> actual length is larger than measured')
	print('Type one, by reference diffusivity')
	print('1st team : {}cm'.format(team_1st.delta_h_1))
	print('2nd team : {}cm'.format(team_2nd.delta_h_1))
	print('3rd team : {}cm'.format(team_3rd.delta_h_1))
	print('4th team : {}cm'.format(team_4th.delta_h_1))
	print('5th team : {}cm'.format(team_5th.delta_h_1))
	print('Type two, by regression intercept')
	print('1st team : {}cm'.format(team_1st.delta_h_2))
	print('2nd team : {}cm'.format(team_2nd.delta_h_2))
	print('3rd team : {}cm'.format(team_3rd.delta_h_2))
	print('4th team : {}cm'.format(team_4th.delta_h_2))
	print('5th team : {}cm'.format(team_5th.delta_h_2))
	
	print('')
	print('-------------------------------------')
	print('---------------1st team--------------')
	print('-------------------------------------')
	team_1st.print_condition()
	team_1st.print_result()
	team_1st.print_error()
	team_1st.Unstst_analysis(team_1st.D_exp)
	
	print('')
	print('-------------------------------------')
	print('---------------2nd team--------------')
	print('-------------------------------------')
	team_2nd.print_condition()
	team_2nd.print_result()
	team_2nd.print_error()
	team_2nd.Unstst_analysis(team_2nd.D_exp)
	
	print('')
	print('-------------------------------------')
	print('---------------3rd team--------------')
	print('-------------------------------------')
	team_3rd.print_condition()
	team_3rd.print_result()
	team_3rd.print_error()
	team_3rd.Unstst_analysis(team_3rd.D_exp)
	
	print('')
	print('-------------------------------------')
	print('---------------4th team--------------')
	print('-------------------------------------')
	team_4th.print_condition()
	team_4th.print_result()
	team_4th.print_error()
	team_4th.Unstst_analysis(team_4th.D_exp)
	
	print('')
	print('-------------------------------------')
	print('---------------5th team--------------')
	print('-------------------------------------')
	team_5th.print_condition()
	team_5th.print_result()
	team_5th.print_error()
	team_5th.Unstst_analysis(team_5th.D_exp)
	'''

	#unsteady state analysis error result comparision
	
	#steady state error
	x_list = np.array([1, 2, 3, 4, 5])
	name_tag = ['1st team', '2nd team', '3rd team', '4th team', '5th team']
	y_list = np.array([team_1st.D_exp, team_2nd.D_exp, team_3rd.D_exp, team_4th.D_exp, team_5th.D_exp])
	ref_list = np.array([team_1st.D_ref, team_2nd.D_ref, team_3rd.D_ref, team_4th.D_ref, team_5th.D_ref])
	y_list = 100*(y_list - ref_list)/ref_list
	y_list = np.abs(y_list)
	
	plt.close()
	plt.xticks(x_list, name_tag)
	plt.bar(x_list, y_list, align = 'center', width = 0.4)
	plt.xlabel('Type')
	plt.ylabel('Error(%)')
	plt.title('Diffusivity error calc by steady state')
	plt.grid()
	plt.ylim(0, 50)
	plt.show()
	
	y_list = np.round_(y_list, 2)
	
	print('1st team : {}%'.format(y_list[0]))
	print('2nd team : {}%'.format(y_list[1]))
	print('3rd team : {}%'.format(y_list[2]))
	print('4th team : {}%'.format(y_list[3]))
	print('5th team : {}%'.format(y_list[4]))
	
	#unsteady state error
	x_list = np.array([1, 2, 3, 4, 5])
	name_tag = ['1st team', '2nd team', '3rd team', '4th team', '5th team']
	y_list = np.array([0.1298, 0.1697, 0.1564, 0.114, 0.1695])
	ref_list = np.array([team_1st.D_ref, team_2nd.D_ref, team_3rd.D_ref, team_4th.D_ref, team_5th.D_ref])
	y_list = 100*(y_list - ref_list)/ref_list
	y_list = np.abs(y_list)
	
	plt.close()
	plt.xticks(x_list, name_tag)
	plt.bar(x_list, y_list, align = 'center', width = 0.4)
	plt.xlabel('Type')
	plt.ylabel('Error(%)')
	plt.title('Diffusivity error calc by unsteady state')
	plt.grid()
	plt.ylim(0, 50)
	plt.show()
	
	y_list = np.round_(y_list, 2)
	
	print('1st team : {}%'.format(y_list[0]))
	print('2nd team : {}%'.format(y_list[1]))
	print('3rd team : {}%'.format(y_list[2]))
	print('4th team : {}%'.format(y_list[3]))
	print('5th team : {}%'.format(y_list[4]))
	
	print('Average : {}cm^2/s'.format(np.average(np.array([0.1298, 0.1697, 0.1564, 0.114, 0.1695]))))

'''	
reference
(1). https://webbook.nist.gov/cgi/cbook.cgi?ID=C67641&Mask=4&Type=ANTOINE&Plot=on
(2). introduction to chemical engineering thermodynamics eighth edition, J.M. smith, Appendix B, Table B.1
(3). https://www.bnl.gov/magnets/staff/gupta/cryogenic-data-handbook/Section6.pdf
(4). http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=Acetone
(5). Perry's Handbook 7th edition Table 2-30
(6). Perry’s chemical engineers’ handbook seventh edition, 374p, table2-371 Diffusivities of pairs of gases and vapors (1atm)
(7). Fundamentals of momentum, heat and mass transfer fifth edition, 410p, Table 24.3
(8). Fundamentals of momentum, heat and mass transfer fifth edition, 416p, Table 24.4
(9). https://www.sigmaaldrich.com/chemistry/solvents/acetone-center.html
'''
