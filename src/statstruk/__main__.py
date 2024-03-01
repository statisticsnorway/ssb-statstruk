"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Statstruk."""


if __name__ == "__main__":
    main(prog_name="statstruk")  # pragma: no cover
