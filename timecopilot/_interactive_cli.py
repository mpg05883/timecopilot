import asyncio
import contextlib
import io
import signal
import sys
from pathlib import Path

import logfire
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status

from .agent import AsyncTimeCopilot, TimeCopilot

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

app = typer.Typer(
    name="timecopilot",
    help="🚀 TimeCopilot - Your GenAI Forecasting Agent",
    rich_markup_mode="rich",
)

console = Console()


class InteractiveTimeCopilot:
    """Interactive TimeCopilot session manager."""

    def __init__(self, llm: str = "openai:gpt-4o-mini", use_async: bool = True):
        self.llm = llm
        self.use_async = use_async
        self.agent: TimeCopilot | AsyncTimeCopilot | None = None
        self.console = Console()
        self.session_active = False
        self.interrupted = False
        self.current_task: asyncio.Task | None = None

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.interrupted = True

        # Cancel current task if running
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.console.print(
                "\n[yellow]Cancelling current operation... Press Ctrl+C again to force quit.[/yellow]"
            )
        else:
            self.console.print(
                "\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]"
            )

    @contextlib.contextmanager
    def _capture_prints(self):
        """Capture print statements and format them nicely."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Process captured output
            stdout_content = stdout_capture.getvalue().strip()
            stderr_content = stderr_capture.getvalue().strip()

            if stdout_content:
                # Format as subdued info
                for line in stdout_content.split("\n"):
                    if line.strip():
                        self.console.print(f"[dim]  → {line}[/dim]")

            if stderr_content:
                # Format as subdued warning
                for line in stderr_content.split("\n"):
                    if line.strip():
                        self.console.print(f"[dim yellow]  ⚠ {line}[/dim yellow]")

    async def _run_cancellable_task(
        self, coro, status_msg: str, show_progress: bool = True
    ):
        """Run a coroutine as a cancellable task with status and progress."""
        try:
            # Create and store the task
            self.current_task = asyncio.create_task(coro)

            if show_progress:
                # Show progress with periodic updates
                result = await self._run_with_progress(self.current_task, status_msg)
            else:
                # Simple status without progress
                with (
                    Status(status_msg, console=self.console),
                    self._capture_prints(),
                ):
                    result = await self.current_task

            return result

        except asyncio.CancelledError:
            self.console.print("\n[yellow]Operation cancelled by user.[/yellow]")
            return None
        finally:
            self.current_task = None

    async def _run_with_progress(self, task, base_msg: str):
        """Run task with progress indicators and cancellation support."""
        progress_msgs = [
            f"{base_msg}",
            f"{base_msg} (loading models...)",
            f"{base_msg} (processing data...)",
            f"{base_msg} (running inference...)",
            f"{base_msg} (finalizing results...)",
        ]

        msg_index = 0

        with self._capture_prints():
            with Status(progress_msgs[0], console=self.console) as status:
                while not task.done():
                    try:
                        # Wait for task completion or timeout
                        result = await asyncio.wait_for(
                            asyncio.shield(task), timeout=3.0
                        )
                        return result
                    except asyncio.TimeoutError:
                        # Update progress message
                        msg_index = (msg_index + 1) % len(progress_msgs)
                        status.update(progress_msgs[msg_index])

                        # Show that we can cancel
                        if msg_index == 1:
                            self.console.print(
                                "[dim]  → Press Ctrl+C to cancel operation[/dim]"
                            )
                        continue

                # Task completed
                return await task

    def _run_sync_with_interrupt_check(self, func, *args, **kwargs):
        """Run a sync function with periodic interrupt checking."""
        import threading
        import time

        result = [None]
        exception = [None]
        finished = [False]

        def worker():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                finished[0] = True

        # Start the work in a separate thread
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

        # Check for interrupts while waiting
        while not finished[0]:
            if self.interrupted:
                self.console.print("\n[yellow]Operation interrupted by user.[/yellow]")
                # Note: We can't actually cancel the sync operation,
                # but we can stop waiting for it
                return None
            time.sleep(0.1)

        if exception[0]:
            raise exception[0]

        return result[0]

    def _create_agent(self) -> TimeCopilot | AsyncTimeCopilot:
        """Create a new agent instance."""
        if self.use_async:
            return AsyncTimeCopilot(llm=self.llm)
        else:
            return TimeCopilot(llm=self.llm)

    def _print_welcome(self):
        """Print welcome message and instructions."""
        welcome_text = """
# 👋 Hi there! I'm TimeCopilot, your time series forecasting companion!

I'm here to help you understand your data and predict the future. Just talk to me 
naturally - no complex commands needed! Think of me as your analytical partner who 
specializes in time series.

## 🤔 **What can we explore together?**
- **Forecast your data**: "Can you predict what happens next with this dataset?"
- **Discover patterns**: "Are there any unusual spikes or anomalies in my data?"
- **Visualize insights**: "Show me a plot of the trends you found"
- **Compare approaches**: "Which model works best - seasonal naive or ARIMA?"
- **Ask about the future**: "What should I expect for the next quarter?"

## 💭 **Just talk to me like this:**
- "I have sales data in this file, can you forecast the next 6 months?"
- "Something seems off with my server metrics - detect any anomalies?"
- "Plot this time series: https://example.com/data.csv"
- "What patterns do you see in my energy consumption data?"
- "How confident are you about next week's predictions?"
- "Compare Chronos and TimesFM models for my dataset"

## 🎯 **Some conversation starters:**
- "Forecast this dataset: /path/to/sales_data.csv"
- "Analyze anomalies in s3://bucket/server-metrics.csv"
- "What will my website traffic look like next month?"
- "Show me a plot of the forecast vs actual values"
- "Which forecasting model gives the most accurate results?"

Ready to dive into your data? Just tell me what you'd like to explore! 🚀
        """

        panel = Panel(
            Markdown(welcome_text),
            title="[bold blue]Welcome to TimeCopilot[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _print_help(self):
        """Print help information."""
        help_text = """
# 💡 Let me help you get the most out of our conversation!

## **🗣️ How to talk with me:**
Just speak naturally! I understand conversational requests, so feel free to ask me 
anything about your time series data. No need for specific commands - I'm designed to 
understand what you want from natural language.

## **💬 Great conversation examples:**
- "I need to forecast next quarter's sales using this dataset: /path/to/sales.csv"
- "Can you spot any weird patterns in my server data?"
- "What's the best model for predicting seasonal retail data?"
- "Show me a visualization of the forecast you just created"
- "How reliable are these predictions for planning purposes?"
- "What should I expect for user engagement next month?"
- "Compare the performance of different forecasting models"
- "Are there any outliers I should be concerned about?"

## **📊 What I can analyze:**
- **CSV files** with columns: unique_id (series name), ds (dates), y (values)
- **Remote data** from URLs (just paste the link!)
- **Parquet files** for larger datasets
- **Multiple time series** in a single file

## **🤖 Models I can use:**
- **Quick & reliable**: SeasonalNaive, AutoARIMA, AutoETS, Prophet
- **Advanced options**: ADIDA, Theta, CrostonClassic, ZeroModel
- **State-of-the-art**: Chronos, TimesFM, and other foundation models
- I'll recommend the best one for your specific data!

## **🚪 Session controls:**
- Type `help` anytime to see this message
- Say `exit`, `quit`, or `bye` when you're done

Remember: I'm here to have a conversation about your data. Ask me anything! 🚀
        """

        panel = Panel(
            Markdown(help_text),
            title="[bold green]How to Chat with TimeCopilot[/bold green]",
            border_style="green",
        )
        self.console.print(panel)

    async def _process_command_async(self, user_input: str) -> bool:
        """Process user command asynchronously."""
        user_input = user_input.strip().lower()

        # Handle exit commands
        if user_input in ["exit", "quit", "bye"]:
            return False

        # Handle help
        if user_input in ["help", "?"]:
            self._print_help()
            return True

        # Check if agent is ready for queries
        if self.agent and self.agent.is_queryable():
            try:
                result = await self._run_cancellable_task(
                    self.agent.query(user_input),
                    "[bold blue]TimeCopilot is thinking...[/bold blue]",
                )

                if result is None:  # Cancelled
                    return True

                # Display result
                response_panel = Panel(
                    result.output,
                    title="[bold cyan]TimeCopilot Response[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
                self.console.print(response_panel)

            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            # Handle initial data loading and forecasting
            await self._handle_initial_command(user_input)

        return True

    def _process_command_sync(self, user_input: str) -> bool:
        """Process user command synchronously."""
        user_input = user_input.strip().lower()

        # Handle exit commands
        if user_input in ["exit", "quit", "bye"]:
            return False

        # Handle help
        if user_input in ["help", "?"]:
            self._print_help()
            return True

        # Check if agent is ready for queries
        if self.agent and self.agent.is_queryable():
            try:
                with (
                    Status(
                        "[bold blue]TimeCopilot is thinking... "
                        "(Press Ctrl+C to interrupt)[/bold blue]",
                        console=self.console,
                    ),
                    self._capture_prints(),
                ):
                    result = self._run_sync_with_interrupt_check(
                        self.agent.query, user_input
                    )

                if result is None:  # Interrupted
                    return True

                # Display result
                response_panel = Panel(
                    result.output,
                    title="[bold cyan]TimeCopilot Response[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
                self.console.print(response_panel)

            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            # Handle initial data loading and forecasting
            self._handle_initial_command_sync(user_input)

        return True

    async def _handle_initial_command(self, user_input: str):
        """Handle initial commands that require data loading."""
        # Extract file path from command
        file_path = self._extract_file_path(user_input)

        if file_path:
            try:
                if not self.agent:
                    self.agent = self._create_agent()

                # Determine the type of analysis requested
                analysis_type = self._detect_analysis_type(user_input)

                if analysis_type == "anomaly_detection":
                    # For anomaly detection, pass the full user input as query
                    result = await self._run_cancellable_task(
                        self.agent.analyze(df=file_path, query=user_input),
                        "[bold red]Loading data and detecting anomalies...[/bold red]",
                    )

                    if result is None:  # Cancelled
                        return
                elif analysis_type == "visualization":
                    # For visualization, first load data with minimal analysis
                    result = await self._run_cancellable_task(
                        self.agent.analyze(
                            df=file_path, query="load data only, no analysis needed"
                        ),
                        "[bold green]Loading data...[/bold green]",
                    )

                    if result is None:  # Cancelled
                        return

                    # Now use the query agent to handle the plot request
                    plot_result = await self._run_cancellable_task(
                        self.agent.query(user_input),
                        "[bold green]Generating visualization...[/bold green]",
                    )

                    if plot_result is None:  # Cancelled
                        return

                    # Create a combined response
                    combined_response = (
                        f"Data loaded successfully.\n\n{plot_result.output}"
                    )

                    # Update the result output
                    result.output.user_query_response = combined_response
                else:  # forecasting (default)
                    # Determine if specific model is requested
                    model_name = self._extract_model_name(user_input)
                    query = f"Use {model_name} model" if model_name else None

                    status_msg = (
                        "[bold blue]Loading data and generating forecast...[/bold blue]"
                    )
                    if model_name:
                        status_msg = f"[bold blue]Loading data and running {model_name}...[/bold blue]"

                    result = await self._run_cancellable_task(
                        self.agent.analyze(df=file_path, query=query), status_msg
                    )

                    if result is None:  # Cancelled
                        return

                # Display results
                result.output.prettify(
                    self.console,
                    features_df=result.features_df,
                    eval_df=result.eval_df,
                    fcst_df=result.fcst_df,
                    anomalies_df=getattr(result, "anomalies_df", None),
                )

                # Success message based on analysis type
                if analysis_type == "anomaly_detection":
                    self.console.print(
                        "\n[bold red]✅ Data loaded and anomaly detection complete![/bold red]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the anomalies, request plots, or run forecasts.[/dim]"
                    )
                elif analysis_type == "visualization":
                    self.console.print(
                        "\n[bold green]✅ Data loaded and visualization complete![/bold green]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the data, detect anomalies, or run forecasts.[/dim]"
                    )
                else:
                    self.console.print(
                        "\n[bold blue]✅ Data loaded and forecast complete![/bold blue]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the forecast, request plots, or detect anomalies.[/dim]"
                    )

            except Exception as e:
                self.console.print(f"[bold red]Error loading data:[/bold red] {e}")
        else:
            self.console.print(
                "[bold yellow]Please provide a file path or URL to get started.[/bold yellow]"
            )
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("[dim]  forecast /path/to/data.csv[/dim]")
            self.console.print("[dim]  detect anomalies in /path/to/data.csv[/dim]")
            self.console.print("[dim]  plot /path/to/data.csv[/dim]")

    def _handle_initial_command_sync(self, user_input: str):
        """Handle initial commands synchronously."""
        # Extract file path from command
        file_path = self._extract_file_path(user_input)

        if file_path:
            try:
                if not self.agent:
                    self.agent = self._create_agent()

                # Determine the type of analysis requested
                analysis_type = self._detect_analysis_type(user_input)

                if analysis_type == "anomaly_detection":
                    with (
                        Status(
                            "[bold red]Loading data and detecting anomalies... "
                            "(Press Ctrl+C to interrupt)[/bold red]",
                            console=self.console,
                        ),
                        self._capture_prints(),
                    ):
                        # For anomaly detection, pass the full user input as query
                        result = self._run_sync_with_interrupt_check(
                            self.agent.analyze, df=file_path, query=user_input
                        )

                    if result is None:  # Interrupted
                        return
                elif analysis_type == "visualization":
                    with (
                        Status(
                            "[bold green]Loading data... "
                            "(Press Ctrl+C to interrupt)[/bold green]",
                            console=self.console,
                        ),
                        self._capture_prints(),
                    ):
                        # For visualization, first load data with minimal analysis
                        result = self._run_sync_with_interrupt_check(
                            self.agent.analyze,
                            df=file_path,
                            query="load data only, no analysis needed",
                        )

                    if result is None:  # Interrupted
                        return

                    with (
                        Status(
                            "[bold green]Generating visualization... "
                            "(Press Ctrl+C to interrupt)[/bold green]",
                            console=self.console,
                        ),
                        self._capture_prints(),
                    ):
                        # Now use the query agent to handle the plot request
                        plot_result = self._run_sync_with_interrupt_check(
                            self.agent.query, user_input
                        )

                    if plot_result is None:  # Interrupted
                        return

                    # Create a combined response
                    combined_response = (
                        f"Data loaded successfully.\n\n{plot_result.output}"
                    )

                    # Update the result output
                    result.output.user_query_response = combined_response
                else:  # forecasting (default)
                    # Determine if specific model is requested
                    model_name = self._extract_model_name(user_input)
                    query = f"Use {model_name} model" if model_name else None

                    status_msg = (
                        "[bold blue]Loading data and generating forecast... "
                        "(Press Ctrl+C to interrupt)[/bold blue]"
                    )
                    if model_name:
                        status_msg = (
                            f"[bold blue]Loading data and running {model_name}... "
                            "(Press Ctrl+C to interrupt)[/bold blue]"
                        )

                    with (
                        Status(status_msg, console=self.console),
                        self._capture_prints(),
                    ):
                        result = self._run_sync_with_interrupt_check(
                            self.agent.analyze, df=file_path, query=query
                        )

                    if result is None:  # Interrupted
                        return

                # Display results
                result.output.prettify(
                    self.console,
                    features_df=result.features_df,
                    eval_df=result.eval_df,
                    fcst_df=result.fcst_df,
                    anomalies_df=getattr(result, "anomalies_df", None),
                )

                # Success message based on analysis type
                if analysis_type == "anomaly_detection":
                    self.console.print(
                        "\n[bold red]✅ Data loaded and anomaly detection complete![/bold red]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the anomalies, request plots, or run forecasts.[/dim]"
                    )
                elif analysis_type == "visualization":
                    self.console.print(
                        "\n[bold green]✅ Data loaded and visualization complete![/bold green]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the data, detect anomalies, or run forecasts.[/dim]"
                    )
                else:
                    self.console.print(
                        "\n[bold blue]✅ Data loaded and forecast complete![/bold blue]"
                    )
                    self.console.print(
                        "[dim]You can now ask questions about the forecast, request plots, or detect anomalies.[/dim]"
                    )

            except Exception as e:
                self.console.print(f"[bold red]Error loading data:[/bold red] {e}")
        else:
            self.console.print(
                "[bold yellow]Please provide a file path or URL to get started.[/bold yellow]"
            )
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("[dim]  forecast /path/to/data.csv[/dim]")
            self.console.print("[dim]  detect anomalies in /path/to/data.csv[/dim]")
            self.console.print("[dim]  plot /path/to/data.csv[/dim]")

    def _detect_analysis_type(self, user_input: str) -> str:
        """Detect the type of analysis requested from user input."""
        user_input_lower = user_input.lower()

        # Check for anomaly detection keywords
        anomaly_keywords = [
            "detect anomalies",
            "anomaly detection",
            "find anomalies",
            "anomalies",
            "outliers",
            "detect outliers",
            "find outliers",
            "unusual",
            "abnormal",
            "irregular",
        ]

        for keyword in anomaly_keywords:
            if keyword in user_input_lower:
                return "anomaly_detection"

        # Check for visualization keywords
        viz_keywords = [
            "plot",
            "chart",
            "graph",
            "visualize",
            "visualization",
            "show",
            "display",
            "draw",
        ]

        for keyword in viz_keywords:
            if keyword in user_input_lower:
                return "visualization"

        # Default to forecasting
        return "forecasting"

    def _extract_file_path(self, user_input: str) -> str | None:
        """Extract file path from user input."""
        # Look for common patterns
        words = user_input.split()

        for i, word in enumerate(words):
            # Check if word looks like a file path or URL
            if (
                word.endswith(".csv")
                or word.endswith(".parquet")
                or word.startswith("http")
                or word.startswith("https")
                or "/" in word
                or "\\" in word
            ):
                return word

            # Check if previous word was "forecast" and this could be a path
            if i > 0 and words[i - 1] in ["forecast", "load", "analyze"]:
                if Path(word).exists() or word.startswith("http"):
                    return word

        return None

    def _extract_model_name(self, user_input: str) -> str | None:
        """Extract model name from user input."""
        # Common model names to look for
        model_names = [
            "seasonal naive",
            "seasonalnaive",
            "arima",
            "autoarima",
            "ets",
            "autoets",
            "prophet",
            "theta",
            "adida",
            "croston",
            "zero",
            "zeromodel",
            "historic",
            "historicaverage",
        ]

        user_lower = user_input.lower()

        for model in model_names:
            if model in user_lower:
                # Convert to proper format
                if model in ["seasonal naive", "seasonalnaive"]:
                    return "SeasonalNaive"
                elif model in ["arima", "autoarima"]:
                    return "AutoARIMA"
                elif model in ["ets", "autoets"]:
                    return "AutoETS"
                elif model == "prophet":
                    return "Prophet"
                elif model == "theta":
                    return "Theta"
                elif model == "adida":
                    return "ADIDA"
                elif model == "croston":
                    return "CrostonClassic"
                elif model in ["zero", "zeromodel"]:
                    return "ZeroModel"
                elif model in ["historic", "historicaverage"]:
                    return "HistoricAverage"

        return None

    async def run_async(self):
        """Run the interactive session asynchronously."""
        self._print_welcome()
        self.session_active = True

        try:
            while self.session_active:
                # Reset interrupted flag
                self.interrupted = False

                try:
                    user_input = Prompt.ask(
                        "\n[bold blue]TimeCopilot[/bold blue]", default=""
                    )
                except KeyboardInterrupt:
                    self.console.print(
                        "\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]"
                    )
                    continue

                if not user_input.strip():
                    continue

                # Check if interrupted during processing
                if self.interrupted:
                    continue

                should_continue = await self._process_command_async(user_input)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print(
                "\n[bold blue]👋 Thanks for using TimeCopilot! See you next time![/bold blue]"
            )

    def run_sync(self):
        """Run the interactive session synchronously."""
        self._print_welcome()
        self.session_active = True

        try:
            while self.session_active:
                # Reset interrupted flag
                self.interrupted = False

                try:
                    user_input = Prompt.ask(
                        "\n[bold blue]TimeCopilot[/bold blue]", default=""
                    )
                except KeyboardInterrupt:
                    self.console.print(
                        "\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]"
                    )
                    continue

                if not user_input.strip():
                    continue

                # Check if interrupted during processing
                if self.interrupted:
                    continue

                should_continue = self._process_command_sync(user_input)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print(
                "\n[bold blue]👋 Thanks for using TimeCopilot! See you next time![/bold blue]"
            )


@app.command()
def chat(
    llm: str = typer.Option(
        "openai:gpt-4o-mini",
        "--llm",
        "-l",
        help="LLM to use for the agent (e.g., 'openai:gpt-4o-mini', 'anthropic:claude-3-haiku')",
    ),
    async_mode: bool = typer.Option(
        True, "--async/--sync", help="Use async mode for better responsiveness"
    ),
):
    """
    🚀 Start an interactive TimeCopilot session.

    This opens a chat-like interface where you can:
    - Load datasets and generate forecasts
    - Ask questions about predictions
    - Detect anomalies
    - Generate plots
    - Use specific forecasting models
    """
    session = InteractiveTimeCopilot(llm=llm, use_async=async_mode)

    if async_mode:
        asyncio.run(session.run_async())
    else:
        session.run_sync()


@app.command()
def forecast(
    file_path: str = typer.Argument(..., help="Path to CSV file or URL"),
    llm: str = typer.Option(
        "openai:gpt-4o-mini", "--llm", "-l", help="LLM to use for forecasting"
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model to use (e.g., 'SeasonalNaive', 'AutoARIMA')",
    ),
    h: int = typer.Option(
        None, "--horizon", "-h", help="Forecast horizon (number of periods)"
    ),
    freq: str = typer.Option(
        None, "--frequency", "-f", help="Data frequency (e.g., 'D', 'M', 'H')"
    ),
    query: str = typer.Option(
        None, "--query", "-q", help="Additional query or instructions"
    ),
):
    """
    📊 Generate a forecast for a dataset (one-shot mode).

    This is a quick way to generate forecasts without entering interactive mode.
    """
    console = Console()

    try:
        with Status("[bold blue]Generating forecast...[/bold blue]", console=console):
            agent = TimeCopilot(llm=llm)

            # Build query with model preference if specified
            full_query = []
            if model:
                full_query.append(f"Use the {model} model")
            if query:
                full_query.append(query)

            final_query = ". ".join(full_query) if full_query else None

            result = agent.forecast(df=file_path, h=h, freq=freq, query=final_query)

        # Display results
        result.output.prettify(
            console,
            features_df=result.features_df,
            eval_df=result.eval_df,
            fcst_df=result.fcst_df,
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show TimeCopilot version information."""
    console.print("[bold blue]TimeCopilot Interactive CLI[/bold blue]")
    console.print("Version: 1.0.0")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
