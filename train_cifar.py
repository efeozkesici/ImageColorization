import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer
from data_cifar import TrainImageFolder
from model import Color_model

original_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.image_dir, original_transform)

    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Build the models
    if torch.cuda.is_available():
        model=nn.DataParallel(Color_model()).cuda()
    else:
        model=Color_model()
    #model.load_state_dict(torch.load('../model/models/model-171-216.ckpt'))
    encode_layer=NNEncLayer()
    boost_layer=PriorBoostLayer()
    nongray_mask=NonGrayMaskLayer()
    # Loss and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)
    

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, img_ab) in enumerate(data_loader):
            # Set mini-batch dataset
            if torch.cuda.is_available():
                images = images.unsqueeze(1).float().cuda()
            else:
                images = images.unsqueeze(1).float()
            img_ab = img_ab.float()
            encode,max_encode=encode_layer.forward(img_ab)
            if torch.cuda.is_available():
                targets=torch.Tensor(max_encode).long().cuda()
            else:
                targets=torch.Tensor(max_encode).long()
            #print('set_tar',set(targets[0].cpu().data.numpy().flatten()))
            if torch.cuda.is_available():
                boost=torch.Tensor(boost_layer.forward(encode)).float().cuda()
                mask=torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()
            else:
                boost=torch.Tensor(boost_layer.forward(encode)).float()
                mask=torch.Tensor(nongray_mask.forward(img_ab)).float()
            boost_nongray=boost*mask
            outputs = model(images)#.log()
            if torch.cuda.is_available():
                output=outputs[0].cpu().data.numpy()
            else:
                output=outputs[0].data.numpy()
            out_max=np.argmax(output,axis=0)
            #print('set',set(out_max.flatten()))
            loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
            #loss=criterion(outputs,targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_epochs, i+1, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save( {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join( args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = 'model/', help = 'path for saving trained models')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    parser.add_argument('--image_dir', type = str, default = 'cifar_imgs_final/', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for printing log info')
    parser.add_argument('--save_step', type = int, default = 216, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 256)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
