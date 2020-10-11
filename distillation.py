import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time

class pseudo_EL():
	def __init__(self):
		self.x_array = np.array([0, 0.019, 0.0721, 0.0966, 0.1238, 0.1661, 0.2337, 0.2608, 0.3273, 0.3965, 0.5079, 0.5198, 0.5732, 0.6763, 0.7472, 0.8943])
		self.y_array = np.array([0, 0.17, 0.3891, 0.4375, 0.4704, 0.5089, 0.5445, 0.558, 0.5826, 0.6122, 0.6564, 0.6599, 0.6841, 0.7385, 0.7815, 0.8943])
		self.R = 0.9
		self.x_last = 0.1287
		self.y_last = 0.5329
		self.x0 = 0.2653
		self.n = 3
		self.eff = None
		
	def EL_find_y(self, x):
		y_array = self.y_array
		x_array = self.x_array
		for i in range(len(x_array)):
			if x < x_array[i]:
				result = (y_array[i]-y_array[i-1])/(x_array[i]-x_array[i-1])*(x-x_array[i])+y_array[i]
				return result

	def EL_find_x(self, y):
		y_array = self.y_array
		x_array = self.x_array
		for i in range(len(y_array)):
			if y < y_array[i]:
				result = (x_array[i]-x_array[i-1])/(y_array[i]-y_array[i-1])*(y-y_array[i])+x_array[i]
				return result
			
	def OL_find_y(self, x):
		R = self.R
		x_D = self.y_last
		return R*x/(1+R)+x_D/(1+R)

	def OL_find_x(self, y):
		R = self.R
		x_D = self.y_last
		return (y-x_D/(1+R))*(1+R)/R
	
	def find_y_by_eff(self, x, eff):
		del_y = eff*(self.EL_find_y(x)-self.OL_find_y(x))
		result = self.OL_find_y(x)+del_y
		return result

	def Rectifying(self):
		n = self.n
		R = self.R
		y_last = self.y_last
		x0 = self.x0
		
		eff = np.arange(0.05,0.4+0.001,0.001)
		diff = np.zeros_like(eff)
		
		for i in range(len(eff)):
			xn = np.array([x0])
			yn = np.array([self.find_y_by_eff(x0, eff[i])])
			for j in range(n):
				temp = self.OL_find_x(yn[j])
				xn = np.hstack([xn, np.array([temp])])
				yn = np.hstack([yn, np.array([self.find_y_by_eff(temp, eff[i])])])
			diff[i] = yn[n]-y_last
			if i>=1:
				if diff[i-1]*diff[i]<0:
					self.eff = (eff[i]+eff[i-1])/2
		plt.close()
		plt.plot(eff,diff)
		plt.xlabel('Efficiency')
		plt.ylabel('y_last(theo)-y_last(exp)')
		plt.title('efficiency vs fraction difference(theo-exp)')
		plt.grid()
		plt.plot([0.05, 0.4], [0, 0])
		plt.show()
	
	def plot_pseudo_EL(self):
		x_array = np.arange(0.05, 0.7+0.001, 0.001)
		pseudo_EL_array = np.zeros_like(x_array)
		OL_array = np.zeros_like(x_array)
		det = np.zeros_like(x_array)
		OL_start_point = False
		
		for i in range(len(x_array)):
			pseudo_EL_array[i] = self.find_y_by_eff(x_array[i], self.eff)
			OL_array[i] = self.OL_find_y(x_array[i])
			det[i] = self.x0 - x_array[i]
			
			if i>=1 and OL_start_point == False:
				if det[i-1]*det[i]<0:
					OL_start_point = i
		
		plt.close()
		plt.plot(self.x_array, self.y_array, label = 'EL')
		plt.plot(x_array, pseudo_EL_array, label = 'pseudo EL')
		plt.plot(x_array[OL_start_point:], OL_array[OL_start_point:], label = 'OL')
		plt.legend()
		plt.grid()
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('EL, pseudo EL, OL for rectifying')
		plt.show()
		
if __name__ == '__main__':
	Mydist = pseudo_EL()
	Mydist.Rectifying()
	print(Mydist.eff)
	Mydist.plot_pseudo_EL()
