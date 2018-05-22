
import numpy as np
import matplotlib.pyplot as plt

train = [100, 300, 400, 500, 700, 1000, 5000]

NLL_orig_mean = [8.5,10.9,11.4,11,13.9,14.8,15.2]
NLL_proposed_mean = [9.4,10.6,11.4,12.3,12.6,12.9,14.4]
NLL_orig_var = [0.8,2,2.6,1.8,6.1,6.5,7.1]
NLL_proposed_var = [0.9,1.3,2.2,6.5,1.1,3.9,5.5]

#plt.figure()
plt.errorbar(train, NLL_orig_mean, yerr=NLL_orig_var)
plt.errorbar(train, NLL_proposed_mean, yerr=NLL_proposed_var)
plt.title("Original & not Original")
plt.show()
