import asyncio
from typing import Any, Optional
from wandb.sdk.wandb_run import Run


class OrderedWandbLogger:
    def __init__(self, wandb_run: Run):
        self.wandb_run = wandb_run
        self.reserved_indexes = set()
        self.logged_indexes = set()
        self.additional_logs = {}
        self.custom_step = "rapidata/step"
        self.wandb_run.define_metric("rapidata/*", step_metric=self.custom_step)    

    def reserve_log_index(self, index: int | None = None, additional_logs: dict[str, int | float] | None = None) -> int:
        """
        Reserve a log index that will be used later.
        If index is None, automatically assigns the next highest available index.
        Returns the reserved index.
        If additional_logs are provided, they will be logged at the same step as the evaluation.
        """
        if index is None:
            if self.reserved_indexes:
                assigned_index = max(self.reserved_indexes) + 1
            else:
                assigned_index = 0
        else:
            assigned_index = index
        
        self.reserved_indexes.add(assigned_index)
        self.additional_logs[assigned_index] = additional_logs or {}
        if additional_logs:
            for key in additional_logs.keys():
                self.wandb_run.define_metric(key, step_metric=self.custom_step)

        return assigned_index
    
    async def log_at_index(self, index: int, metrics: dict[str, Any]) -> None:
        """
        Log metrics at a specific reserved index.
        Will wait to log until all earlier indexes have been logged.
        The index becomes the wandb step.
        """
        log_dict = {self.custom_step: index, **metrics, **self.additional_logs[index]}

        self.wandb_run.log(log_dict)
        self.logged_indexes.add(index)

    async def wait_for_all_logs(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until all reserved indexes have been logged.
        Returns True if all logs were completed, False if timeout occurred.
        """
        start_time = asyncio.get_event_loop().time()
        
        while self.logged_indexes != self.reserved_indexes:
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                return False
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
        
        return True
