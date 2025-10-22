import train


def main() -> None:
    print("Loading data...")
    data = train.train('../dataset')
    print("Loading complete.")

    return


if __name__ == '__main__':
    main()
