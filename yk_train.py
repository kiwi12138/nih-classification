from datasets import *
import sys
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
if __name__=='__main__':

    transform = A.Compose([A.Resize(224,224), A.Normalize(mean=0.5, std=0.5),
                                ToTensorV2()])
    #dataset
    data_dir=r'E:\data\nih-chest'
    XRayTest_dataset = XRaysTestDataset(data_dir, transform=transform)
    XRayTrain_dataset = XRaysTrainDataset(data_dir, transform=transform)
    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset,
                                                               [int(len(XRayTrain_dataset) * train_percentage),
                                                                len(XRayTrain_dataset) - int(
                                                                    len(XRayTrain_dataset) * train_percentage)])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True,persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128,  shuffle=False)
    test_loader = torch.utils.data.DataLoader(XRayTest_dataset,batch_size=128,  shuffle=False)
    #model
    from torchvision.models import resnet50
    model=resnet50(pretrained=True)
    model.conv1.weight = Parameter(torch.mean(model.conv1.weight, dim=1, keepdim=True), requires_grad=True)
    model.fc=nn.Linear(2048,14)

    model.cuda()
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad], "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]},
    ]
    ce=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(param_dicts,lr=1e-4,weight_decay=1e-4)
    best_val=0
    best_test=0
    for i in range(100):
        model.train()
        for iter,(image,label) in enumerate(train_loader):
            assert image.size(1)==1
            assert torch.sum(label).item()==image.size(0)
            label=torch.argmax(label,-1).cuda()
            image=image.cuda()
            logits=model(image)
            loss=ce(logits,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sys.stdout.write(f'\r epoch {i} iter {iter} loss {loss.item()}')
        model.eval()

        preds=[]
        gts=[]
        for (image,label) in val_loader:
            assert image.size(1)==1
            assert torch.sum(label).item()==image.size(0)
            label=torch.argmax(label,-1).cuda()
            image=image.cuda()
            with torch.no_grad():
                logits=model(image)
            pred=torch.argmax(logits,-1)
            preds.append(pred)
            gts.append(label)
        preds=torch.cat(preds,0)
        gts=torch.cat(gts,0)
        val_acc=torch.eq(preds,gts).float().mean().item()

        preds=[]
        gts=[]
        for (image,label) in test_loader:
            assert image.size(1)==1
            assert torch.sum(label).item()==image.size(0)
            label=torch.argmax(label,-1).cuda()
            image=image.cuda()
            with torch.no_grad():
                logits=model(image)
            pred=torch.argmax(logits,-1)
            preds.append(pred)
            gts.append(label)
        preds=torch.cat(preds,0)
        gts=torch.cat(gts,0)
        test_acc=torch.eq(preds,gts).float().mean().item()

        best_val=best_val if best_val>val_acc else val_acc
        best_test=best_test if best_test>test_acc else test_acc
        print(f'\n Epoch {i},val/test acc:{val_acc} / {test_acc}, BEST val/test acc:{best_val} / {best_test}')



