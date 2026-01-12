#!/usr/bin/env python3
"""
Awesome AI and ML - Interactive CLI

A collection of AI and ML concept and paper implementations built from scratch.
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()


class Implementation:
    """Represents a single AI/ML implementation."""

    def __init__(self, name, path, description, command):
        self.name = name
        self.path = path
        self.description = description
        self.command = command


# Define all available implementations
IMPLEMENTATIONS = [
    Implementation(
        name="Mini Neural Network",
        path="mini-neural-network",
        description="A 2-layer neural network from scratch. Demonstrates forward/backward propagation, gradient descent, and binary classification on non-linear data.",
        command=["uv", "run", "test_nn.py"],
    ),
    Implementation(
        name="Mini Bigram LM",
        path="mini_lm",
        description="A simple Bigram Language Model implemented from scratch using NumPy. It learns the probability of a word given the previous word.",
        command=["uv", "run", "main.py"],
    ),
    Implementation(
        name="Decision Tree",
        path="decision_tree",
        description="This project implements a Decision Tree Classifier using pure NumPy. It's a classic supervised learning algorithm that makes decisions by splitting data based on information gain.",
        command=["uv", "run", "main.py"],
    ),
]


def show_banner():
    """Display the main banner and description."""
    banner = Text()
    banner.append("Awesome AI & ML\n", style="bold cyan")
    banner.append("A Collection of AI/ML Implementations from Scratch", style="dim")

    console.print(Panel(banner, box=box.DOUBLE, border_style="cyan", padding=(1, 2)))

    console.print()
    console.print(
        "[dim]This monorepo contains educational implementations of various AI and ML concepts,[/dim]"
    )
    console.print(
        "[dim]algorithms, and papers. Each is built from the ground up to demonstrate core principles.[/dim]"
    )
    console.print()


def show_implementations_table():
    """Display a table of all available implementations."""
    table = Table(
        title="Available Implementations",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )

    table.add_column("#", style="cyan", width=4, justify="right")
    table.add_column("Name", style="green bold", width=25)
    table.add_column("Description", style="white")

    for idx, impl in enumerate(IMPLEMENTATIONS, 1):
        table.add_row(str(idx), impl.name, impl.description)

    console.print(table)
    console.print()


def run_implementation(impl: Implementation):
    """Run the selected implementation."""
    console.print()
    console.print(
        Panel(f"[bold green]Running:[/bold green] {impl.name}", border_style="green")
    )
    console.print()

    # Change to the implementation directory
    impl_path = Path(__file__).parent / impl.path

    try:
        # Run the command in the implementation directory
        subprocess.run(impl.command, cwd=impl_path, check=True)

        console.print()
        console.print(f"[bold green]✓[/bold green] {impl.name} completed successfully!")

    except subprocess.CalledProcessError as e:
        console.print()
        console.print(f"[bold red]✗[/bold red] Error running {impl.name}")
        console.print(f"[red]Exit code: {e.returncode}[/red]")
    except FileNotFoundError:
        console.print()
        console.print(
            f"[bold red]✗[/bold red] Could not find the implementation at {impl_path}"
        )


def show_menu():
    """Display the interactive menu and handle user input."""
    while True:
        show_banner()
        show_implementations_table()

        # Build choices
        choices = [str(i) for i in range(1, len(IMPLEMENTATIONS) + 1)]
        choices.append("q")

        prompt_text = f"[cyan]Select an implementation (1-{len(IMPLEMENTATIONS)}) or 'q' to quit:[/cyan]"
        choice = Prompt.ask(prompt_text, choices=choices, default="q")

        if choice.lower() == "q":
            console.print()
            console.print("[yellow]Thanks for exploring! Happy learning! 👋[/yellow]")
            console.print()
            break

        # Run the selected implementation
        impl_index = int(choice) - 1
        run_implementation(IMPLEMENTATIONS[impl_index])

        # Ask if user wants to continue
        console.print()
        continue_choice = Prompt.ask(
            "[cyan]Press Enter to return to menu, or 'q' to quit[/cyan]",
            choices=["", "q"],
            default="",
        )

        if continue_choice.lower() == "q":
            console.print()
            console.print("[yellow]Thanks for exploring! Happy learning! 👋[/yellow]")
            console.print()
            break

        # Clear screen for next iteration
        console.clear()


def main():
    """Main entry point."""
    try:
        show_menu()
    except KeyboardInterrupt:
        console.print()
        console.print("\n[yellow]Interrupted by user. Goodbye! 👋[/yellow]")
        console.print()
        sys.exit(0)


if __name__ == "__main__":
    main()
