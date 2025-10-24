from train import train
from model_io import save_model, load_model


def main() -> None:
    print("PICTVRA QVANTICA\n")

    model, category = train(num_images_per_category=500)
    save_model(model, category)

    return


if __name__ == '__main__':
    main()
