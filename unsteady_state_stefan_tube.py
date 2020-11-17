#Numerical method for solving problem
#Boundary condition : P(x=0) = Ps, P(x = L) = 0, P(t = 0) = 0 

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

R=82.06
MA = 58.08
T = 46+273
rho = 57.6214/(0.233955**(1+(1-T/507.803)**0.254167))/1000

x0 = -5.28 		#cm
D = 0.152 		#cm^2/s
P_s = 10**(4.42448 -(1312.253/(T-32.445)))*0.9869		#B.C where x = 0
P_e = 0		#B.C where x = L
P = 0.988

n_x = 50
dx_0 = abs(x0)/n_x

t_end = 3600			#s
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
plt.grid()
plt.xlabel('time(s)')
plt.ylabel('diffusion length(cm)')
plt.title('t vs diffusioon length')
plt.show()
