import asyncio
from typing import Any, Optional
from wandb.sdk.wandb_run import Run


class OrderedWandbLogger:
    def __init__(self, wandb_run: Run):
        self.wandb_run = wandb_run
        self.reserved_indexes = set()
        self.pending_logs = {}  # index -> metrics
        self.next_expected_index = None
        self.lock = asyncio.Lock()
        self.custom_step = "rapidata/step"
        self.wandb_run.define_metric("rapidata/*", step_metric=self.custom_step)    

    def reserve_log_index(self, index: int | None = None) -> int:
        """
        Reserve a log index that will be used later.
        If index is None, automatically assigns the next highest available index.
        Returns the reserved index.
        """
        if index is None:
            # Find the next highest index by taking max of reserved indexes + 1
            if self.reserved_indexes:
                assigned_index = max(self.reserved_indexes) + 1
            else:
                assigned_index = 0
        else:
            assigned_index = index
        
        self.reserved_indexes.add(assigned_index)
        if self.next_expected_index is None or assigned_index < self.next_expected_index:
            self.next_expected_index = assigned_index
        
        return assigned_index
    
    def reserve_log_indexes(self, indexes: list[int]) -> None:
        """Reserve multiple log indexes."""
        for index in indexes:
            self.reserve_log_index(index)
    
    async def log_at_index(self, index: int, metrics: dict[str, Any]) -> None:
        """
        Log metrics at a specific reserved index.
        Will wait to log until all earlier indexes have been logged.
        The index becomes the wandb step.
        """
        async with self.lock:
            if index not in self.reserved_indexes:
                raise ValueError(f"Index {index} was not reserved")
            
            # Store the log data
            self.pending_logs[index] = metrics
            
            # Try to flush any logs that can now be sent
            await self._flush_ready_logs()
    
    async def _flush_ready_logs(self) -> None:
        """Flush all logs that are ready to be sent in order."""
        while (self.next_expected_index is not None and 
               self.next_expected_index in self.pending_logs):
            
            # Get the metrics for the next expected index
            metrics = self.pending_logs.pop(self.next_expected_index)
            
            log_dict = {self.custom_step: self.next_expected_index, **metrics}

            self.wandb_run.log(log_dict)
            
            # Remove from reserved indexes
            self.reserved_indexes.remove(self.next_expected_index)
            
            # Find the next expected index
            self.next_expected_index = min(self.reserved_indexes) if self.reserved_indexes else None
    
    async def wait_for_all_logs(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until all reserved indexes have been logged.
        Returns True if all logs were completed, False if timeout occurred.
        """
        start_time = asyncio.get_event_loop().time()
        
        while self.reserved_indexes:
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                return False
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
        
        return True
