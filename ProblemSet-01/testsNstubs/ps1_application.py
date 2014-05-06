	


def usps():
    ''' performs the usps analysis for assignment 4'''
    
                
        
def outliers_calc():
    ''' outlier analysis for assignment 5'''
    
    d = np.load('banana_data.npz')
    data = d['data']
    label = d['label']
    print 'data: ', data.shape
    print 'label: ', label.shape
    pos_idx = np.where(label == 1)
    neg_idx = np.where(label == -1)
    pos_data = data[:, pos_idx[1]]
    neg_data = data[:, neg_idx[1]]
    print 'pos: ', pos_data.shape
    print 'neg: ', neg_data.shape
    for rate in [0.01, 0.05, 0.1, 0.25]:
    	total_outliers = rate * pos_data.shape[1]
    	outlier_idx = np.random.choice(neg_idx[1], total_outliers, replace=False)
    	print 'outlier_idx: ', len(outlier_idx)
    	outlier_data = data[:, outlier_idx]
    	pos_data_outlier = np.concatenate((pos_data, outlier_data), axis=1)
    	print 'pos_data_outlier: ', pos_data_outlier.shape
    # np.savez_compressed('outliers', var1=var1, var2=var2, ...)

            
def outliers_disp():
    ''' display the boxplots'''
    results = np.load('outliers.npz')
    
            
            

def lle_visualize(dataset='flatroll'):
    ''' visualization of LLE for assignment 6'''
    
    

def lle_noise():
    ''' LLE under noise for assignment 7'''    


