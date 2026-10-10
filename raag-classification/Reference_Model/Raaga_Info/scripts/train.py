import torch

from train_model import train_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    model, train_loader, val_loader, criterion, optimizer = train_model()

    EPOCHS = 20

    best_accuracy = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        correct = 0

        total = 0

        for chroma, labels in train_loader:

            chroma = chroma.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(chroma)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {running_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for chroma, labels in val_loader:

                chroma = chroma.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(chroma)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total

        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            torch.save(model.state_dict(), "best_model.pth")

            print("Best model saved!")

    print("\nTraining Finished!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()