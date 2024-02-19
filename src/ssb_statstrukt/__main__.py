"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Statstrukt."""


if __name__ == "__main__":
    main(prog_name="ssb-statstrukt")  # pragma: no cover
