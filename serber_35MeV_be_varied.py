#DM: Modified code below to varies angle over a 3x5mm target at 2.0cm from Be foil; incident deuteron energy 35 MeV; current 2uA; calculate time to achieve a fluence of 1E14 neutrons

#Python code for calculating neutron flux at a given angle for a given deuteron energy and breakup target (the sample provided is for 40 MeV deuterons on beryllium at 0°, but this can be easily modified).

import numpy as np

def serber_yield(E_n, E_d=40.0, deg=0.0, E_min=1.0, A=69.9E-3, w=26.6, R_E=1.0, R_tht=1.0, R_bg=1.0, R_pe=1.0, rho=1.85, amu=9.0, z=4.0):

	def dEdx(E_d, n, z):
		I = 11.2+11.7*z if z<14 else 52.8+8.71*z
		beta2 = 1.0 - (1.0/(1.0 + (E_d/1.8756E3))**2)
		F = np.log(1.02E6*beta2/(1.0-beta2))-beta2
		S = 0.508*(z*n)*(F-np.log(I))/beta2
		return np.where(S>5, S, 5.0)

	def P_En(E_n, E_d, E_c, R_E=1.0):
		w = R_E*0.385*2.225
		eps = E_d-E_c

		x = eps*w/((E_n-0.5*eps)**2+eps*w)**1.5
		sm = 1.0 + eps/(eps**2+4.0*eps*w)**0.5
		return x/sm

	def P_tht(tht, E_d, E_c, R_tht=1.0):
		tht_0 = R_tht*0.812*np.sqrt(2.225/(E_d-E_c))*(1.0-((E_d-E_c)/8E3))

		x = tht_0/(tht_0**2+tht**2)**1.5
		sm = np.pi/(tht_0*np.sqrt(tht_0**2+np.pi**2))
		return x/sm

	def kalbach_xs(E_d, amu, A=24.7E-3, w=9.7):
		mu = 22.3
		return A*(amu**0.333+1.26)/(1.0+np.exp((mu-E_d)/w))

	def kalbach_a(E_d, A, Z):
		def S_b(A, Z):
			N = A-Z
			s1 = 15.68*(2.0)
			s2 = 28.07*((N+1-Z)**2/(A+2)-(N-Z)**2/A)
			s3 = 18.56*((A+2)**0.667-A**0.667)
			s4 = 33.22*((N+1-Z)**2/(A+2)**1.333-(N-Z)**2/A**1.333)
			s5 = 0.717*((Z+1)**2/A**0.333-Z**2/A**0.333)
			s6 = 1.211*((Z+1)**2/(A+2)-Z**2/A)
			return s1-s2-s3+s4-s5+s6
		eb = 0.5*E_d+S_b(A+1,Z+1)
		ea = E_d+S_b(A,Z)-2.225
		E1 = min((ea, 130.0))
		E3 = min((ea, 41.0))
		return 0.04*(E1*eb/ea)+1.8E-6*(E1*eb/ea)**3+6.7E-7*0.5*(E3*eb/ea)**4


	def T_lorentz(T, E_d, amu, tht):
		gm = 1.0 + E_d/(931.5*(amu+2.013))
		beta = np.sqrt(1.0-1.0/gm**2)
		return T/(gm*(1.0-beta*np.cos(tht)))

	def compound(E_n, E_d, tht, A=9.0, Z=4.0, R_bg=1.0):
		K, m1, a1, m2, e2 = 4.03E-3, 7.0, 0.3, 14.0, 18.0
		
		tht_cm = np.arctan((1.0 + E_d/(931.5*(2.013+A)))*np.tan((tht-90.0)*np.pi/180.0))+(np.pi/2.0)
		# needs phase shift for tan/arctan to map correctly
		E0 = 0.1+np.sqrt(0.075*(E_d+4.36224))
		E0 = T_lorentz(E0, E_d, A, tht_cm)

		a = 1.1*kalbach_a(E_d, A, Z)
		p = (0.5*a/np.sinh(a))*(np.exp(a*np.cos(tht_cm))+np.exp(-a*np.cos(tht_cm)))

		E_range = np.arange(0, E_d+0.1, 0.1)
		norm = np.trapezoid(np.sinh(np.sqrt(2.0*E_range))*np.exp(-E_range/E0), E_range)
		B = np.sinh(np.sqrt(2.0*E_n))*np.exp(-E_n/E0)/norm

		return R_bg*B*p*K*(np.exp(-0.5*((e2-E_d)/m2)**2)+a1/(1.0+np.exp((e2-E_d)/m1)))

	
	def pre_equilibrium(E_n, E_d, tht, A, Z, R_pe=1.0):		
		K, m1, e1 = 50.3E-3, 6.0, 22.0
		
		tht_cm = np.arctan((1.0 + E_d/(931.5*(2.013+A)))*np.tan((tht-90.0)*np.pi/180.0))+(np.pi/2.0)
		# needs phase shift for tan/arctan to map correctly
		E0 = 0.75+np.sqrt(0.4*(E_d+4.36224))
		E0 = T_lorentz(E0, E_d, A, tht_cm)

		a = 1.8*kalbach_a(E_d, A, Z)
		p = (a/np.sinh(a))*np.exp(a*np.cos(tht_cm))

		E_range = np.arange(0, E_d+0.1, 0.1)
		norm = np.trapezoid(np.sinh(np.sqrt(2.0*E_range))*np.exp(-E_range/E0), E_range)
		B = np.sinh(np.sqrt(2.0*E_n))*np.exp(-E_n/E0)/norm

		return R_pe*B*p*K/(1.0+np.exp((e1-E_d)/m1))


	def tau(E_d, E_inc, n, amu, z):
		if E_d==E_inc:
			return 1.0

		E_range = np.linspace(E_d, E_inc, 100)
		ddx = dEdx(E_range, n, z)

		r0_2_barns = 0.015625
		c1, a1, a2 = 5.643, 1.354, 131.3
		sigma = (np.pi*r0_2_barns*(amu**(1/3.0) + 0.8)**2)*c1*(1.0-np.exp(-E_range/a1))*np.exp(-E_range/a2)
		sigma += kalbach_xs(E_range, amu)

		return np.exp(-np.trapezoid(n*sigma/ddx, E_range))



	n = rho*6.022E-1/amu
	E_c = (z*197/137.0)/(1.57+1.25*amu**0.333)-0.9
	dE, Y = 1.0, (np.zeros(len(E_n)) if E_n.shape else 0.0)
	
	for ed in np.arange((E_min if E_min>E_c+1 else E_c+1), E_d+dE, dE):
		tau_E = tau(ed, E_d, n, amu, z)
		Y += (1.0/1.602E-13)*n*tau_E*kalbach_xs(ed, amu, A, w)*P_En(E_n, ed, E_c, R_E)*P_tht(deg*np.pi/180.0, ed, E_c, R_tht)*dE/dEdx(ed, n, z)
		Y += (1.0/1.602E-13)*n*tau_E*(compound(E_n, ed, deg, amu, z, R_bg)+pre_equilibrium(E_n, ed, deg, amu, z, R_pe))*dE/dEdx(ed, n, z)

	return Y


###############
# MAIN SEQUENCE
###############
if __name__=="__main__":
        E_inc = 35.0 # MeV
        current = 2 # uA
        dE = 1.0 # Energy grid spacing
        x_list = np.arange(0,3,0.25) # target x range (mm)
        y_list = np.arange(0,5,0.25) # target y range (mm)
        Total_yield_list_pos = []  # list of yields to average over
        for x in x_list:
                for y in y_list:
                        tht = np.arctan(np.sqrt(x**2 + y**2)/2.0) * 180/np.pi
                        E_range = np.arange(dE, E_inc+dE, dE)
                        total_yield = 0
                        for e in enumerate(E_range):
                                total_yield += np.average(serber_yield(E_range, E_inc, tht))
                        Total_yield_list_pos.append(total_yield)
        avg_yield = sum(Total_yield_list_pos)/len(Total_yield_list_pos)
        fluence_per_hour = avg_yield * (current * 60 * 60) * 0.08  # fluence after 1-hr run at 2uA, 0.08 sr for 3x5mm target
        time_to_1e14 = 1.0e+14 / fluence_per_hour * 60
        print("Time to 1E14 total fluence is {0:.0f} minutes.".format(time_to_1e14))

