from __future__ import annotations

from unstructured.partition.auto import partition


def main():

    elements = partition(
        filename='',
    )
    print(elements)


if __name__ == '__main__':
    main()
