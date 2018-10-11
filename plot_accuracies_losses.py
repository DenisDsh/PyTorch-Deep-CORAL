import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    path_adapt = './logs/log_adaptation.pth'
    path_no_adapt = './logs/log_no_adaptation.pth'

    # Load logs
    log_adapt = torch.load(path_adapt)
    log_no_adapt = torch.load(path_no_adapt)

    # compute mean for each epoch
    adapt = {
        'classification_loss': torch.FloatTensor(log_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    no_adapt = {
        'classification_loss': torch.FloatTensor(log_no_adapt['classification_loss']).mean(dim=1).numpy(),
        'coral_loss': torch.FloatTensor(log_no_adapt['CORAL_loss']).mean(dim=1).numpy(),
        'source_accuracy': torch.FloatTensor(log_no_adapt['source_accuracy']).mean(dim=1).numpy(),
        'target_accuracy': torch.FloatTensor(log_no_adapt['target_accuracy']).mean(dim=1).numpy()
    }

    # Add the first 0 value
    adapt['target_accuracy'] = np.insert(adapt['target_accuracy'], 0, 0)
    no_adapt['target_accuracy'] = np.insert(no_adapt['target_accuracy'], 0, 0)
    adapt['source_accuracy'] = np.insert(adapt['source_accuracy'], 0, 0)
    no_adapt['source_accuracy'] = np.insert(no_adapt['source_accuracy'], 0, 0)

    plt.gca().set_color_cycle(['blue', 'green', 'red', 'm'])

    axes = plt.gca()
    axes.set_ylim([0, 1.1])

    l1, = plt.plot(adapt['target_accuracy'], label="test acc. w/ coral loss", marker='*')
    l2, = plt.plot(no_adapt['target_accuracy'], label="test acc. w/o coral loss", marker='.')

    l3, = plt.plot(adapt['source_accuracy'], label="training acc. w/ coral loss", marker='^')
    l4, = plt.plot(no_adapt['source_accuracy'], label="training acc. w/o coral loss", marker='+')

    plt.legend(handles=[l1, l2, l3, l4], loc=4)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()

    fig_acc.savefig('accuracies.pdf', dpi=1000)

    # Classification loss and CORAL loss for training w/ CORAL loss
    plt.gca().set_color_cycle(['red', 'blue'])

    axes = plt.gca()
    axes.set_ylim([0, 0.5])

    l5, = plt.plot(adapt['classification_loss'], label="classification loss", marker='.')
    l6, = plt.plot(adapt['coral_loss'], label="coral loss", marker='*')

    plt.legend(handles=[l5, l6], loc=1)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()
    fig_acc.savefig('losses_adapt.pdf', dpi=1000)

    # CORAL distance for training w/o CORAL loss (lambda = 0)
    plt.gca().set_color_cycle(['blue'])
    l7, = plt.plot(no_adapt['coral_loss'], label="distance w/o coral loss", marker='.')

    plt.legend(handles=[l7], loc=2)

    fig_acc = plt.gcf()
    plt.show()
    plt.figure()
    fig_acc.savefig('coral_loss_no_adapt.pdf', dpi=1000)


if __name__ == '__main__':
    main()