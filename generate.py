import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt


def caluculate_entropy(p):

	S = -p * math.log(p,2)
	return S


def compute_totalentropy(Countarray,n,d):

	Entropyarray = np.ndarray(shape=d)

	for i in range(d):
		Entropyarray[i] = 0
	index = 0
	for item in Countarray:
		sume = 0
		for p in item:
			p = p/n
			if p == 0:
				tmp = 0
			else:
				tmp = caluculate_entropy(p)
			sume = sume + tmp
		Entropyarray[index] = sume
		index = index + 1

	return Entropyarray


def rank_dimension(Entropyarray,n,d):

	Rankarray = np.ndarray(shape=d)

	for i in range(d):
		Rankarray[i] = 0

	Sortedarray = np.sort(Entropyarray)

	for i in range(len(Entropyarray)):
		for j in range(len(Sortedarray)):
			flag = 1
			if Entropyarray[i] == Sortedarray[j]:
				flag = 0
				Rankarray[i] = j + 1
			if flag == 0:
				break
			else:
				pass


	Rankarray = np.transpose(Rankarray)
	return Rankarray


def rank_points(PointEntropy,n,d):

	Rankpoints = np.zeros(shape=n)

	Sortedarray = np.sort(PointEntropy)

	for i in range(len(PointEntropy)):
		for j in range(len(Sortedarray)):
			flag = 1
			if PointEntropy[i] == Sortedarray[j]:
				Rankpoints[i] = j + 1
				flag = 0
			if flag == 0:
				break
			else:
				pass
	return Rankpoints 


def  get_Entropy(dataT,Countarray,Parlen,Pararray,n,d):
	
	IndividualEntropy = np.zeros(shape=(d,n))
	tmp = 0
	for item in dataT:
		for j in range(len(item)):
			for i in range(Parlen-1):
				if item[j] >= Pararray[i] and item[j] < Pararray[i+1]:
					S = Countarray[tmp][i]
					P = S/n
					Entropy = caluculate_entropy(P)
					IndividualEntropy[tmp][j] = Entropy 
		#print Countarray[tmp]
		#print 
		tmp = tmp + 1

	IndividualEntropy = np.transpose(IndividualEntropy)

	return IndividualEntropy


def Metric(item,Rankarray):
	tmp = np.multiply(item,Rankarray)
	return tmp


def rank_pointentropy(IndividualEntropy,Rankarray,n,d):

	PointEntropy = np.zeros(shape=n)
	tmp = 0
	for item in IndividualEntropy:
		s = Metric(item,Rankarray)
		PointEntropy[tmp] = np.sum(s)
		tmp = tmp + 1		

	return PointEntropy


def getoriginalentropy(IndividualEntropy,n,d):

	Get_OriginalPointentropy = np.zeros(shape=n)
	tmp = 0
	for item in IndividualEntropy:
		Get_OriginalPointentropy[tmp] = np.sum(item)
		tmp = tmp + 1

	Get_OriginalPointentropysum = np.sum(Get_OriginalPointentropy)
	return Get_OriginalPointentropysum
	


def  iterate(dataT,data,n,d):

	DMin = float('+inf')
	DMax = float('-inf')
	for item in dataT:
		a = np.amax(item)
		b = np.amin(item)
		if b < DMin :
			DMin = round(b,8)
		if DMax < a :
			DMax = round(a,8)


	#print DMin
	#print
	#print DMax
	#print

	Parlen = DMax - DMin
	#Parlen = int(Parlen)
	Parlen = 6


	Pararray = np.ndarray(shape=Parlen)

	for i in range(Parlen):
		Pararray[i] = 0
		

	for i in range(Parlen):
		tmp = round(np.random.uniform(DMin,DMax) , 8)
		if tmp in Pararray:
			i = i - 1
		else:
			Pararray[i] = tmp

	Pararray = np.sort(Pararray)
	Pararray[0] = DMin - 1
	Pararray[Parlen-1] = DMax + 1
	#print 'This is Pararray'
	#print Pararray

	Countarray = np.zeros(shape=(d,Parlen-1))

	#print Countarray
	#print
	tmp = 0
	for item in dataT:
		for element in item:
			for i in range(Parlen-1):
				if element >= Pararray[i] and element < Pararray[i+1]:
					Countarray[tmp][i] = Countarray[tmp][i] + 1 
		#print Countarray[tmp]
		#print 
		tmp = tmp + 1

	#print Countarray

	#print 
	#print Parlen-1

	Entropyarray = compute_totalentropy(Countarray,n,d)
	#print Entropyarray

	Rankarray = rank_dimension(Entropyarray,n,d)
	#print Rankarray

	IndividualEntropy = get_Entropy(dataT,Countarray,Parlen,Pararray,n,d)
	#print IndividualEntropy

	PointEntropy = rank_pointentropy(IndividualEntropy,Rankarray,n,d)
	#print PointEntropy

	Rankpoints = rank_points(PointEntropy,n,d)
	#print Rankpoints

	Get_OriginalPointentropysum = getoriginalentropy(IndividualEntropy,n,d)
	#print Get_OriginalPointentropysum

	return Get_OriginalPointentropysum


if __name__ == '__main__':

	n = 10000
	"""d = 3

	data = np.random.normal(0,1,(n,d))
	dataT = np.transpose(data)
	iterate(dataT,data,n,d)
	"""

	Dimarray = []
	Entarray = []

	Dimarray.append(0)
	Entarray.append(0)

	for d in range(1000):
		if d == 0:
			pass
		else:
			data = np.random.normal(0,1,(n,d))
			dataT = np.transpose(data)
			Get_sum = iterate(dataT,data,n,d)
			Dimarray.append(d)
			Entarray.append(Get_sum)
		print d

	plt.plot(Dimarray,Entarray)
	plt.xlabel('Dimension')
	plt.ylabel('Entropy')
	plt.show()

	
