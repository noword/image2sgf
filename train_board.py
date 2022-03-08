from gogame_dataset import RandomBoardDataset
from model import get_board_model, get_transform
import torch
import os
from engine import train_one_epoch, evaluate
import utils


def main(pth_name, max_hands=361, batch_size=5, device=None):
    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_board_model()
    if os.path.exists(pth_name):
        model.load_state_dict(torch.load(pth_name, map_location=device))
    model.to(device)

    _dataset = RandomBoardDataset(1000, max_hands=max_hands, transforms=get_transform(train=True))
    indices = torch.randperm(len(_dataset)).tolist()
    dataset = torch.utils.data.Subset(_dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(_dataset, indices[-50:])
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 20
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    for epoch in range(num_epochs):
        # torch.cuda.empty_cache()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), pth_name)
        _dataset.initseed()

    print("That's it!")


if __name__ == '__main__':
    # my graphics card only has 4G memory, batch_size had to be set to 1
    main('weiqi_board.pth', batch_size=1)
