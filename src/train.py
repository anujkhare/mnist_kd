import os
import pathlib
import tensorboardX


def forward_once(model, batch, loss=None):
    images, labels = batch
    images = images.cuda()
    labels = labels.cuda()

    preds = model(images)
    error = None
    if loss is not None:
        error = loss(preds, labels)
    return preds, error


def predict(model, dataloader, first_n=-1):
    list_all_preds = []
    for iteration, next_batch in enumerate(dataloader):
        preds, _ = forward_once(model, next_batch)
        preds = torch.argmax(preds, dim=1).data.cpu().numpy()
        list_all_preds += list(preds)

        if first_n > 0 and iteration + 1 >= first_n:
            break

    return np.array(list_all_preds)


def validate_model(model, dataloader, loss, n_iters=10, plot=True):
    assert n_iters > 0
    model.train(False)

    total_acc = 0
    total_error = 0
    for iteration, next_batch in enumerate(dataloader):
        preds, error = forward_once(model, next_batch, loss)
        preds = torch.argmax(preds, dim=1).data.cpu().numpy()
        acc = np.mean(preds == next_batch[1].numpy())

        total_error += error.data.cpu().numpy()
        total_acc += acc

        if iteration + 1 >= n_iters:
            break

    mean_acc = total_acc / (iteration + 1)
    mean_error = total_error / (iteration + 1)

    if plot:
        visualize_batch(next_batch)
        print('Preds:', preds)
        plt.show()

    model.train(True)

    return mean_error, mean_acc


def train(model, n_epochs=100, val_every=1000, log_every=100):
    """
    NOTE: DEPENDS ON MANY GLOBAL VARIABLES!
    """
    model.train(True)
    epochs = np.arange(n_epochs)

    total_iters = 0
    for epoch in epochs:
        for iteration, next_batch in enumerate(dataloader_train):
            total_iters += 1
            optimizer.zero_grad()
            preds, error = forward_once(model, next_batch, loss)
            error.backward()
            optimizer.step()

            # calculate acc
            preds = torch.argmax(preds, dim=1).data.cpu().numpy()
            acc_train = np.mean(preds == next_batch[1].numpy())

            if iteration % log_every == 0:
                writer.add_scalar('train.loss', error.data.cpu().numpy(), total_iters)
                writer.add_scalar('train.acc', acc_train, total_iters)

            if iteration % val_every == 0:
                err_val, acc_val = validate_model(model, dataloader_val, loss)

                print('Epoch: {}, Iteration: {}'.format(epoch, total_iters))
                print('Loss: {:.4f}, Accuracy: {:.2f}%'.format(err_val, acc_val * 100))
                writer.add_scalar('val.loss', err_val, total_iters)
                writer.add_scalar('val.acc', acc_val, total_iters)

        # Save every epoch
        torch.save(model.state_dict(), path_to_weights / '{}.pt'.format(epoch))