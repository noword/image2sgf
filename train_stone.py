from img2sgf import *
from img2sgf.stone_dataset import *
from img2sgf.random_transforms import get_stone_transform
import time
import torch
import os

try:
    from apex import amp
except ImportError:
    amp = None


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                    print_freq, apex=False, model_ema=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    if model_ema:
        model_ema.update_parameters(model)


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=''):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: {log_suffix}'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(f'{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    return metric_logger.acc1.global_avg


def main(pth_name, theme_path=None, data_path=None, epochs=10, batch_size=5, num_workers=1, device=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_stone_model()
    if os.path.exists(pth_name):
        model.load_state_dict(torch.load(pth_name, map_location=device))
    model.to(device)

    _dataset = StoneDataset(theme_path=theme_path,
                            data_path=data_path,
                            transforms=get_stone_transform(True))

    indices = torch.randperm(len(_dataset)).tolist()
    test_num = int(len(_dataset) * 0.1)
    dataset = torch.utils.data.Subset(_dataset, indices[:-test_num])
    dataset_test = torch.utils.data.Subset(_dataset, indices[-test_num:])
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=1)

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.0001,
                                momentum=0.9,
                                weight_decay=0.0001,
                                )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_aac1 = .0
    for epoch in range(epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, 10)
        lr_scheduler.step()
        aac1 = evaluate(model, criterion, data_loader_test, device)
        if aac1 > best_aac1:
            best_aac1 = aac1
            print(f'New best aac1 {best_aac1}')
            torch.save(model.state_dict(), pth_name)


if __name__ == '__main__':
    main(pth_name='stone.pth',
         theme_path='./themes',
         data_path='./stone_data',
         epochs=20,
         batch_size=32,
         num_workers=2)
