import train


def main() -> None:
    for i in [5, 10, 15, 20, 25, 30, 31]:
        print(f"PCA Components: {i}")
        train.train(num_pca_components=i)
        print()

    return


if __name__ == '__main__':
    main()
