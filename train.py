from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from model import MobileNet, MobileNetV3_Small
from dataset import My_dataset
from tqdm import tqdm
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train():

    learning_rate = 0.01
    shape = [640,480,3]
    # Batch_size = 32
    # num_workers = 3
    # Epoch = 100
    Batch_size = 5
    num_workers = 0
    Epoch = 3
    cuda = torch.cuda.is_available()
    #model = MobileNet()
    model = MobileNetV3_Small()
    if cuda:
        model = model.cuda().train()
    else:
        model = model.train()
    #print(model)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
    train_dataset = My_dataset(shape,"train.txt","train_label.txt")
    val_dataset = My_dataset(shape,"val.txt","val_label.txt")
    train_loader = DataLoader(train_dataset,batch_size=Batch_size, num_workers=num_workers, pin_memory=False)#pin_memory=True
    val_loader = DataLoader(val_dataset,batch_size=Batch_size, num_workers=num_workers, pin_memory=False)
    
    steps_train = len(train_loader)
    steps_val = len(val_loader)

    total_loss = 0
    val_loss = 0
    try:
        for epoch in range(Epoch):
            total_loss = 0
            val_loss = 0
            print("start training")
            with tqdm(total = steps_train, desc=f'Epoch{epoch+1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
                for iteration,(img,label) in enumerate(train_loader):

                    if cuda:
                        img = img.cuda()
                        label = label.cuda()
                    #forward
                    pred = model(img)
                    loss = loss_function(pred,label)
                    total_loss += loss.item() 
                    #backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #display
                    pbar.set_postfix(**{'train_loss' : total_loss / (iteration + 1), 
                                        'lr'         : get_lr(optimizer)})
                    pbar.update(1)
            print("start validation")
            with tqdm(total = steps_val, desc=f'Epoch{epoch+1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
                for iteration,(img,label) in enumerate(val_loader):
                    with torch.no_grad():
                        if cuda:
                            img = img.cuda()
                            label = label.cuda()
                        #forward
                        pred = model(img)
                        loss = loss_function(pred,label)
                        val_loss += loss.item()

                    #display
                    pbar.set_postfix(**{'val_total_loss' : val_loss / (iteration + 1), 
                                        'lr'         : get_lr(optimizer)})
                    pbar.update(1)

            #learning_scheduler
            lr_scheduler.step()
            torch.save(model.state_dict(),f'v3-{val_loss:.3f}.pt')
    except Exception as e:
        print(e)
        torch.save(model.state_dict(), f'v3-{val_loss:.3f}.pt')

if __name__=="__main__":
    train()