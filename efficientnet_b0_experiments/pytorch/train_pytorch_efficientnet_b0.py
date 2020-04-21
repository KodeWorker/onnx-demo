# -*- coding: utf-8 -*-
import os
import torch
from torch import nn, optim
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from tqdm import trange

def accuracy(batch, model):
    x, y = batch
    x, y = x.to(device), y.to(device)
    # y_pred = (model.predict(x)).type(torch.FloatTensor)
    y_pred = (model.forward(x))
    correct = torch.sum(torch.argmax(y_pred, dim=1) == y).item() / len(y)
    return correct

if __name__ == "__main__":
    
    RANDOM_SEED = 5566
    NUM_CLASSES = 2
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    IMG_SIZE = 224
    
    torch.manual_seed(RANDOM_SEED)
    
    model = EfficientNet.from_name("efficientnet-b0", override_params={"num_classes": NUM_CLASSES})
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
                                            transforms.Resize((IMG_SIZE, IMG_SIZE)), # resize image
                                            transforms.ToTensor(), # to (0, 1)
                                            transforms.Normalize(mean, std) # normalize image
                                            ])
    
    datadir = r"D:\Datasets\NB-CONN\divided"
    traindir = os.path.join(datadir, "train")
    
    train_data = datasets.ImageFolder(traindir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    
    t = trange(1, EPOCHS+1, leave=True)
    for epoch in t:
        
        batch = None
        running_loss = 0
        
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            loss.backward()
            optimizer.step()
            
            if not batch:
                batch = (inputs, labels)
            else:
                batch = (torch.cat((batch[0], inputs)), torch.cat((batch[1], labels)))
            running_loss += loss.item()
        
        with torch.no_grad():
            acc = accuracy(batch, model)            
        loss = running_loss/len(train_loader)
        t.set_description("train loss: {:.4E}, train acc: {:.2f}%".format(loss, acc*100))
        t.refresh()
    
    model_path = 'pytorch_efficientnet_b0_weights.pth'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)