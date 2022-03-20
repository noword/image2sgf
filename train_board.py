from gogame_dataset import RandomBoardDataset
from model import get_board_model, get_transform
import torch
import os
from engine import train_one_epoch, evaluate
import utils


def main(pth_name, hands_num=(1, 361), batch_size=5, num_workers=1, data_size=1000, device=None):
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_board_model()
    if os.path.exists(pth_name):
        model.load_state_dict(torch.load(pth_name, map_location=device), strict=False)
    model.to(device)

    dataset = RandomBoardDataset(initvar=data_size, hands_num=hands_num, transforms=get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=utils.collate_fn)

    dataset_test = RandomBoardDataset(initvar=50, hands_num=hands_num, transforms=get_transform(train=True))
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.0001)
    optimizer = torch.optim.SGD(params, lr=0.0002, momentum=0.9, weight_decay=0.0001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(16, 22), gamma=0.1)

    num_epochs = 26
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    for epoch in range(num_epochs):
        # torch.cuda.empty_cache()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), pth_name)
        dataset.initseed()
        dataset_test.initseed()

    print("That's it!")


if __name__ == '__main__':
    # my graphics card only has 4G memory, batch_size had to be set to smaller
    main('board.pth', hands_num=(1, 361), batch_size=1, num_workers=2, data_size=500)
