import sys
from pylint import lint
from argparse import ArgumentError


def main():

    THRESHOLD = 6

    if len(sys.argv) < 2:
        raise ArgumentError("Module to evaluate needs to be the first argument")

    run = lint.Run([sys.argv[1], "-d C0301"], do_exit=False)
    score = run.linter.stats['global_note']

    if score < THRESHOLD:
        print("your score failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
