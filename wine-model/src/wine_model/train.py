from .data import load_data
from .model import train_model


def main():
    data = load_data()
    model, accuracy = train_model(data)
    print(f"Model trained with accuracy: {accuracy: .2f}")


if __name__ == "__main__":
    main()
