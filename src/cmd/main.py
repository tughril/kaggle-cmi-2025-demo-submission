#!/usr/bin/env python3
"""
Main CLI entry point using typer.
"""

import typer
from typing_extensions import Annotated

app = typer.Typer(help="ML Template CLI")


@app.command()
def hello(
    name: Annotated[str, typer.Argument(help="Name to greet")] = "World",
    count: Annotated[
        int, typer.Option("--count", "-c", help="Number of greetings")
    ] = 1,
):
    """Say hello to someone."""
    for _ in range(count):
        typer.echo(f"Hello {name}!")


@app.command()
def train(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to train")
    ] = "default",
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs")
    ] = 10,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
):
    """Train a machine learning model."""
    if verbose:
        typer.echo(f"Training {model} model for {epochs} epochs...")
    else:
        typer.echo(f"Training {model} model...")

    # TODO: Implement actual training logic
    typer.echo("Training completed!")


@app.command()
def predict(
    input_file: Annotated[str, typer.Argument(help="Input file path")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model to use")
    ] = "default",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output file path")
    ] = "predictions.json",
):
    """Make predictions using a trained model."""
    typer.echo(f"Making predictions on {input_file} using {model} model...")
    typer.echo(f"Predictions saved to {output}")

    # TODO: Implement actual prediction logic


if __name__ == "__main__":
    app()
