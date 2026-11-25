from thermalModel import ThermalModel
import dataLoad
import torch
from torch import nn


def test_function(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0.0
    correct = 0
    total_acc = 0
    total_losses = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(dataLoad.device)
            y = y.to(dataLoad.device)

            y_pred = model(x)
            total_loss += loss_fn(y_pred, y).item()

            predicted = torch.threshold(y_pred.data, 0.5,0)

            correct += (predicted == y).sum().item()
            # print(predicted.cpu(), y.cpu(), (predicted == y).sum().cpu())
            total_acc += y.size(0)
            total_losses += 1

    loss = total_loss / total_losses

    accuracy = 100 * correct / total_acc
    return loss, accuracy


def model_training_function_single_pass(model, dataloader, loss_fn, optimizer):
    model.train()  # set the model to training mode

    total_loss = 0.0
    correct = 0
    total = 0

    for batch, (x, y) in enumerate(dataloader):
        # ensure the training data is on the correct device
        x = x.to(dataLoad.device)
        y = y.to(dataLoad.device)

        # zero the optimiser
        optimizer.zero_grad()
        # genraete the predictions
        y_pred = model(x)

        loss = loss_fn(y_pred, y)  # calculate the loss
        loss.backward()  # update the weights (back propagation)
        optimizer.step()  # set the optimiser

        total_loss += loss.item() * x.size(0)  # add to total loss

        total += y.size(0)

        # if batch % 10 == 0:
        #     print(f"Batch {batch}: Loss {loss}")
    total_loss = total_loss / total
    return total_loss


class ModelTrainer:
    def __init__(self, model, data_loader, name, lr=0.01, epochs=100):
        self.model = model
        self.epochs = epochs
        self.learning_rate = lr
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.data_loader = data_loader
        self.name = name

    def train(self, validation_data_loader=None):
        for epoch in range(self.epochs):
            loss = model_training_function_single_pass(self.model, self.data_loader, self.loss_fn, self.optimizer)
            print("EEEP ", epoch, " loss: ", loss)
            if epoch != 0 and epoch % 50 == 0:
                torch.save(self.model.state_dict(), self.name + "_model_t_" + str(epoch) + ".pth")
                print("saved: " + self.name + "_model_t_" + str(epoch) + ".pth")
                _, train_acc = test_function(self.model, self.data_loader, self.loss_fn)
                print("train accuracy: ", train_acc)
                if validation_data_loader is not None:
                    _, val_acc = test_function(self.model, validation_data_loader, self.loss_fn)
                    print("validation accuracy: ", val_acc)


def load_thermal_model(create=False):
    model = None
    if create:
        model = ThermalModel().to(dataLoad.device)
    else:
        model = ThermalModel()
        model.load_state_dict(torch.load('thermal_model.pth', map_location=torch.device('cpu')))
        # this allows for it to run on a device different than the one it was trained on
        model = model.to(dataLoad.device)
    return model


if __name__ == '__main__':
    dataLoad.loadThermalData()
    thermal_model = load_thermal_model(True)

    model_trainer = ModelTrainer(thermal_model, dataLoad.thermal_train_data_loader,"thermal", 0.0001, 100)
    model_trainer.train(validation_data_loader=dataLoad.thermal_validation_data_loader)
    torch.save(thermal_model.state_dict(), "thermal_model.pth")
    print("done, saved thermal_model.pth")
    _, train_acc = test_function(thermal_model, dataLoad.thermal_train_data_loader, nn.BCELoss())
    print("train accuracy: ", train_acc)
    _, val_acc = test_function(thermal_model, dataLoad.thermal_validation_data_loader, nn.BCELoss())
    print("validation accuracy: ", val_acc)
