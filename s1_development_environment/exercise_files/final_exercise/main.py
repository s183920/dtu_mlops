import click
import torch
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from data import mnist

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=25, help="number of epochs to train for")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print("Learning rate: ", lr)
    print("Epochs: ", epochs)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.to(DEVICE)
    train_set, _ = mnist()
    
    # setup training
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    losses = [None]*epochs
    
    # run training
    for epoch in range(epochs):
        num_correct = 0
        num_data = 0
        running_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_data += len(images)
            num_correct += (pred.argmax(1) == labels).sum().item()
        losses[epoch] = running_loss/len(train_set)
        print("Epoch {}: \t train loss: {} \t accuracy: {}".format(epoch, losses[epoch], num_correct/num_data))
    
    # save model
    torch.save(model, 'trained_model.pt')
    
    # plot loss
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(xlabel = 'Epoch', ylabel = 'Loss', title = 'Training Loss')
    plt.savefig('loss.png')
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    model = model.to(DEVICE)
    _, test_set = mnist()
    
    num_correct = 0
    num_data = 0
    for images, labels in test_set:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        pred = model(images)
        
        num_data += len(images)
        num_correct += (pred.argmax(1) == labels).sum().item()
    print("Test accuracy: {}".format(num_correct/num_data))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
