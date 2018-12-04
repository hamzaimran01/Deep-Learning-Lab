

import matplotlib.pyplot as plt


import numpy as np
with open('testIoU.txt','r') as f:
    
    
    
    lines=f.readlines()
    result_x=[]
    result_y=[]
    
    for x in lines:
        result_x.append(x.split(' ')[0])
        print(result_x)
        result_y.append(x.split(' ')[1])
        print(result_y)
        
            

    f.close()
    plt.plot(result_x,result_y)    
    plt.show()    
    


    

    
