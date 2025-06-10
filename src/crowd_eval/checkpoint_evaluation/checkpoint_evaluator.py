from rapidata import RapidataClient, RapidataOutputManager
from crowd_eval.logger.ordered_wandb_logger import OrderedWandbLogger
from wandb.sdk.wandb_run import Run
from typing import Any, Optional
import asyncio
import threading
from abc import abstractmethod
import json
from rapidata.api_client.exceptions import ApiException
from rapidata.api_client.models.order_state import OrderState
from rapidata.rapidata_client.order.rapidata_results import RapidataResults
from rapidata.rapidata_client.order.rapidata_order import RapidataOrder

class Evaluator:
    def __init__(self, wandb_run: Run, model_name: str | None = None, client_id: str | None = None, client_secret: str | None = None):
        RapidataOutputManager.enable_silent_mode()
        self.client = RapidataClient(client_id=client_id, client_secret=client_secret)
        self.model_name = model_name or "model"
        self.logger = OrderedWandbLogger(wandb_run)
        self.prompts = self._get_prompts()
        self.baseline_prompts = None
        self.baseline_media = None
        
        # Keep track of background tasks to prevent garbage collection
        self._background_tasks: set[Any] = set()
        self._background_loop: Optional[asyncio.AbstractEventLoop] = None
        self._background_thread: Optional[threading.Thread] = None

    def define_baseline(self, image_paths: list[str], prompts: list[str]) -> None:
        if len(image_paths) != len(prompts):
            raise ValueError("Number of images and prompts must be the same")
        self.baseline_prompts = prompts
        self.baseline_media = image_paths

    @abstractmethod
    def evaluate(self, step: int | None = None) -> None:
        pass
    
    @abstractmethod
    def _get_prompts(self) -> dict[int, str]:
        pass

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure we have a background event loop running in a separate thread."""
        if self._background_loop is None or not self._background_loop.is_running():
            def start_background_loop() -> None:
                self._background_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._background_loop)
                self._background_loop.run_forever()
            
            self._background_thread = threading.Thread(
                target=start_background_loop, 
                daemon=True,
                name=f"CheckpointEvaluator-{self.model_name}"
            )
            self._background_thread.start()
            
            # Wait a moment for the loop to start
            import time
            time.sleep(0.1)
    
        # Return the loop (guaranteed to exist now)
        assert self._background_loop is not None
        return self._background_loop

    async def _wait_for_results_async(self, order: RapidataOrder, poll_interval: float = 5.0) -> RapidataResults:
        """Asynchronously wait for order completion and return results."""
        completed_states = [OrderState.COMPLETED, OrderState.PAUSED, OrderState.MANUALREVIEW, OrderState.FAILED]
        
        # Poll for completion without blocking
        while True:
            status = await asyncio.to_thread(order.get_status)
            if status in completed_states:
                break
            await asyncio.sleep(poll_interval)
        
        # Get the final results
        try:
            results_json = await asyncio.to_thread(lambda: order.get_results())
            return results_json
        except (ApiException, json.JSONDecodeError) as e:
            raise Exception(f"Failed to get order results: {str(e)}") from e

    def _schedule_background_evaluation(self, coro, assigned_index: int) -> None:
        """Schedule a background evaluation coroutine to run."""
        # Check if we're in an async context
        try:
            # Try to get the running loop
            loop = asyncio.get_running_loop()
            # We're in an async context, use create_task
            task = loop.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running loop, we're in sync context
            # Create or reuse background thread with event loop
            loop = self._ensure_background_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            self._background_tasks.add(future)
            future.add_done_callback(self._background_tasks.discard)

    def wait_for_all_evaluations(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all background evaluations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all evaluations completed, False if timeout occurred
        """
        return asyncio.run(self.logger.wait_for_all_logs(timeout))

    def __del__(self):
        """Cleanup background thread when evaluator is destroyed."""
        if self._background_loop and self._background_loop.is_running():
            self._background_loop.call_soon_threadsafe(self._background_loop.stop)

