from torchvision.models import alexnet
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import get_loader
import config
from utils import accuracy, Tracker
from coral import coral


def train(model, optimizer, source_loader, target_loader, epoch=0):

    tracker = Tracker()
    model.train()
    tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}

    # Trackers to monitor classification and CORAL loss
    classification_loss_tracker = tracker.track('classification_loss', tracker_class(**tracker_params))
    coral_loss_tracker = tracker.track('CORAL_loss', tracker_class(**tracker_params))

    min_n_batches = min(len(source_loader), len(target_loader))

    tq = tqdm(range(min_n_batches), desc='{} E{:03d}'.format('Training + Adaptation', epoch), ncols=0)

    for _ in tq:

        source_data, source_label = next(iter(source_loader))
        target_data, _ = next(iter(target_loader))  # Unsupervised Domain Adaptation

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()

        out_source = model(source_data)
        out_target = model(target_data)

        classification_loss = F.cross_entropy(out_source, source_label)

        # This is where the magic happens
        coral_loss = coral(out_source, out_target)
        composite_loss = classification_loss + config.lambda_coral * coral_loss

        composite_loss.backward()
        optimizer.step()

        classification_loss_tracker.append(classification_loss.item())
        coral_loss_tracker.append(coral_loss.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(classification_loss=fmt(classification_loss_tracker.mean.value),
                       coral_loss=fmt(coral_loss_tracker.mean.value))


def evaluate(model, data_loader, dataset_name, epoch=0):
    model.eval()

    tracker = Tracker()
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    acc_tracker = tracker.track('accuracy', tracker_class(**tracker_params))

    loader = tqdm(data_loader, desc='{} E{:03d}'.format('Evaluating on %s' % dataset_name, epoch), ncols=0)

    accuracies = []
    with torch.no_grad():
        for target_data, target_label in loader:
            target_data = Variable(target_data)
            target_label = Variable(target_label)

            output = model(target_data)

            accuracies.append(accuracy(output, target_label))

            acc_tracker.append(sum(accuracies)/len(accuracies))
            fmt = '{:.4f}'.format
            loader.set_postfix(accuracy=fmt(acc_tracker.mean.value))


def main():

    source_train_loader = get_loader(name_dataset='amazon', batch_size=config.batch_size, train=True)
    target_train_loader = get_loader(name_dataset='webcam', batch_size=config.batch_size, train=True)

    source_evaluate_loader = get_loader(name_dataset='amazon', batch_size=config.batch_size, train=False)
    target_evaluate_loader = get_loader(name_dataset='webcam', batch_size=config.batch_size, train=False)

    n_classes = len(source_train_loader.dataset.classes)

    # ~ Paper : "We initialized the other layers with the parameters pre-trained on ImageNet"
    # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    model = alexnet(pretrained=True)
    # ~ Paper : The dimension of last fully connected layer (fc8) was set to the number of categories (31)
    model.classifier[6] = nn.Linear(4096, config.n_classes)
    # ~ Paper : and initialized with N(0, 0.005)
    torch.nn.init.normal_(model.classifier[6].weight, mean=0, std=5e-3)

    # Initialize bias to small constant number (http://cs231n.github.io/neural-networks-2/#init)
    model.classifier[6].bias.data.fill_(0.01)

    # ~ Paper : "The learning rate of fc8 is set to 10 times the other layers as it was training from scratch."
    optimizer = torch.optim.SGD([
        {'params':  model.features.parameters()},
        {'params': model.classifier[:6].parameters()},
        # fc8 -> 7th element (index 6) in the Sequential block
        {'params': model.classifier[6].parameters(), 'lr': 10 * config.lr}
    ], lr=config.lr, momentum=config.momentum)  # if not specified, the default lr is used

    for i in range(config.epochs):
        train(model, optimizer, source_train_loader, target_train_loader, i)
        evaluate(model, source_evaluate_loader, 'source', i)
        evaluate(model, target_evaluate_loader, 'target', i)


if __name__ == '__main__':
    main()