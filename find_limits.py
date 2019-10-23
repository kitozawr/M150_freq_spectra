def find_limits(data, method='simple'):
	'''
	Функция находит максимальное и минимальное значение для печати двумерного массива.
	'''
	#Константы
	width = 3
	if method=='simple':
		v_max = np.amax(data)
	elif method=='good':
		#Максимальное среднее по квадрату 3x3.
		y_rest = data.shape[0] % width
		x_rest = data.shape[1] % width
		wv = int(round(width-1)/2)
		wv_1 = wv+1
		ws = width*width
		curr_max = 0.0; new_max = 0.0
		for i in range(wv, data.shape[0]-wv, width):
			for j in range(wv, data.shape[1]-wv, width):
				new_max = np.sum(data[i-wv:i+wv_1,j-wv:j+wv_1])/ws
				if new_max > curr_max:
					curr_max = new_max
		v_max = curr_max	
	else:
		print("ERROR: in find_limits (lumin_proc.py) - unknown keyword for scale maxima search")
		return False		
	v_min = (np.sum(data[:20, :20]) + np.sum(data[-20:, :20]) + np.sum(data[:20,-20:]) + np.sum(data[-20:,-20:]))/1600.0
	return (v_min, v_max)
