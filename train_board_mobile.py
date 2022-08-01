from img2sgf import *
from img2sgf.random_transforms import *
from img2sgf.gogame_dataset import *
from img2sgf.engine import *
import torch
import os

BEST_SCORE_NAME = 'best_mobile.score'
TRAIN_PATH = 'train'
TEST_PATH = 'test'


def main(pth_name, hands_num=(1, 361), batch_size=5, num_workers=1, data_size=10000, device=None):
    dataset = RandomBoardDataset(initvar=data_size, hands_num=hands_num, transforms=get_transform(train=True))
    gen_cache(dataset, TRAIN_PATH)

    dataset = RandomBoardDataset(initvar=100, hands_num=hands_num, transforms=get_transform(train=True))
    gen_cache(dataset, TEST_PATH)

    if device:
        device = torch.device(device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_board_mobile_model()
    if os.path.exists(pth_name):
        model.load_state_dict(torch.load(pth_name, map_location=device), strict=False)
    model.to(device)

    dataset_test = CachedDataset(TEST_PATH)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   collate_fn=utils.collate_fn)
    dataset = CachedDataset(TRAIN_PATH)
    indices = torch.randperm(len(dataset))
    sampler = torch.utils.data.SubsetRandomSampler(indices[:3000])
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=utils.collate_fn,
                                              sampler=sampler)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.0001)
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(16, 22), gamma=0.1)

    num_epochs = 10
    if os.path.exists(BEST_SCORE_NAME):
        best_score = float(open(BEST_SCORE_NAME).read())
    else:
        best_score = -9999
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    for epoch in range(num_epochs):
        # torch.cuda.empty_cache()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_test, device=device)
        score = sum(evaluator.coco_eval['bbox'].stats)
        print(f'current score: {score}, best score: {best_score}')
        if score > best_score:
            best_score = score
            open(BEST_SCORE_NAME, 'w').write(str(best_score))
        torch.save(model.state_dict(), pth_name)

        # dataset.initseed()
        # dataset_test.initseed()

    print(f"That's it! Best score is {best_score}")


if __name__ == '__main__':
    main('board_mobile.pth', hands_num=(1, 361), batch_size=8, num_workers=1, data_size=10000)
