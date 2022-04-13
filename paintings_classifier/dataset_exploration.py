import torch

#balanced dataset
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight                      

def balance_dataset():
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,                              
                                                                sampler = sampler, num_workers=args.workers, pin_memory=True) 

if __name__ == '__main__'    