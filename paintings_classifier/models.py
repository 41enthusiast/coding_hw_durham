from torch import nn
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
import pytorch_lightning as pl


from efficientnet_pytorch import EfficientNet

class FinetunedModel(nn.Module):       
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x    
    def __init__(self, n_classes = 2, freeze_base= True, hidden_size = 512):
        super().__init__()
        
        self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
        internal_embedding_size = self.base_model._fc.in_features
        self.base_model._fc = FinetunedModel.Identity()
        
        #non linear projection head improves the embedding quality 
        self.fc_head = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features= n_classes if n_classes > 2 else 1)
        )
        if freeze_base:
            print("Freezing embeddings")
            for param in self.base_model.parameters():
                param.requires_grad = False
                

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc_head(out)
        return out
