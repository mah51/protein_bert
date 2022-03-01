import seaborn as sn
import matplotlib.pyplot as plt
import pickle


with open('confusion_matrix_dump.pkl', 'rb') as file:
  confusion_matrix = pickle.load(file)

confusion_matrix = confusion_matrix.filter(items=['GO:0005840', 'GO:0003735' ])

plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(confusion_matrix, annot=True)
cfm_plot.figure.savefig("cfm.png")