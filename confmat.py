import numpy as np

def print_confusion_matrix(ergebnis, classes):
	
	n=len(classes)
	m=len(ergebnis)
	
	confusion_matrix=np.zeros((n, n), dtype=int)
	
	for (x,y) in ergebnis:
		confusion_matrix[y,x]+=1
	
	string="Real\t|Predicted\n\t"
	
	for i in range(n):
		string+="|"+str(classes[i])+"\t"
	string+="\n-"
	
	for i in range(n):
		string+="-------+"
	string+="-------\n"
	
	
	for i in range(n):
		string+=str(classes[i])+"\t"
		for j in range(n):
			string+="|"+str(confusion_matrix[i,j])+"\t"
		string+="\n\t"
		for j in range(n):
			percentage=round(confusion_matrix[i,j]/m*100, 1)
			
			string+="|"+str(percentage)+" %\t"
		string+="\n-"
		for i in range(n):
			string+="-------+"
		string+="-------\n"
	
	print(string)

#print_confusion_matrix(test, [1,5,7])
