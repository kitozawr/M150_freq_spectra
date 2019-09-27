import numpy as np

def read_bd_map(bd_map_file):
	"""
	Read breakdown map, as specified in bd_map_file.
	It is assumed that coordinates of breakdowns are sorted by x increasing.
	"""
	#Variables
	bd_mult = [] #Координаты "множественных" пробоев.
	bd_single = [] #Координаты "одиночных" пробоев.

	f=open(bd_map_file)
	lines = f.readlines()
	print("Reading breakdown map...")
	if lines[0] == "Muliple hot spots\n":
		print("File start is OK.")
	single_start_num = lines.index("Separate hot spots\n")
	bd_mult = np.genfromtxt(bd_map_file, skip_header=1, max_rows=single_start_num-1, dtype = 'uint16')
	bd_single = np.genfromtxt(bd_map_file, skip_header=single_start_num+1, dtype = 'uint16')
	print("Bd map has been successfully read.")

	return (bd_mult, bd_single)

def apply_bd_map(data, bd_mult, bd_single):
	"""
	Removes breakdowns, which coordinates are listed in bd_mult ("multiple" breakdowns) and bd_single (single separated hot spots).
	It is assumed that coordinates of breakdowns are sorted by x increasing.
	"""
	#Для множественных пробоев - аппроксимируем пробитый участок линейной зависимостью, исходя из ближайших непробитых точек.
	i_old = 0
	if bd_mult.size != 0:
		for k in range(bd_mult.shape[0]):
			j,i = bd_mult[k]
			if i == i_old and j>j_bottum and j<j_top:
				continue
			elif i < i_old:
				print("ERROR: coordinates are not sorted by i increasing!!!")
				return "apply_bd_map_error"
			else:
				i_old = i; k1 = k
				while k1<bd_mult.shape[0] and bd_mult[k1,1] == i: #Используем, что массив отсортирован по i, а где i одинаково - по j.
					k1 += 1
				j_top = int(bd_mult[k1-1,0])+1; k1=k #Ближайшая непробитая точка сверху.
				while k1>=0 and bd_mult[k1,1] == i:
					k1 -= 1
				j_bottum = int(bd_mult[k1+1,0])-1 #Ближайшая непробитая точка снизу.
				# Коэффициенты линейной аппроксимации.
				A = (float(data[j_top,i]) - float(data[j_bottum,i]))/(float(j_top) - float(j_bottum))
				B = float(data[j_bottum,i]) - A*j_bottum
				data[j_bottum+1:j_top,i] = np.around(A*np.arange(j_bottum+1, j_top)+B)

	#Для одиночных пробоев.
	if bd_single.size != 0:
		for k in range(bd_single.shape[0]):
			j,i = bd_single[k]
			vic_sum = np.sum(data[j-1:j+2,i-1:i+2]) - data[j,i]
			data[j,i] = vic_sum/8.0
	return data
