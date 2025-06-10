import asyncio
import os
import pandas as pd
from crowd_eval.checkpoint_evaluation.checkpoint_evaluator import Evaluator
from wandb.sdk.wandb_run import Run

IMAGE_EVALUATION_RESPONSES_PER_DATAPOINT = 10

class ImageEvaluator(Evaluator):
    def __init__(self, wandb_run: Run, model_name: str | None = None, client_id: str | None = None, client_secret: str | None = None):
        super().__init__(wandb_run, model_name, client_id, client_secret)

    def evaluate(self, image_paths: list[str], step: int | None = None) -> None:
        """
        Fire-and-forget evaluation that will log results to wandb when complete.
        This method returns immediately and evaluation happens in background.
        
        Args:
            image_paths: List of image paths to evaluate
            step: Step number for wandb logging (if None, auto-assigned)
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        if self.baseline_media is not None:
            if not len(image_paths) == len(self.baseline_media):
                raise ValueError("Number of images must match the number of baseline images")
            
        if not self.baseline_media:
            self._check_images(image_paths)
        
        assigned_index = self.logger.reserve_log_index(step)
        
        # Schedule the background evaluation using the base class method
        coro = self._background_evaluate(image_paths, assigned_index)
        self._schedule_background_evaluation(coro, assigned_index)

    def _get_prompts(self) -> dict[int, str]:
        prompts_df = pd.read_json("https://assets.rapidata.ai/image_prompts_4o.json")
        return {row["id"]: row["prompt"] for _, row in prompts_df.iterrows()}

    def _check_images(self, image_paths: list[str]) -> bool:
        invalid_prompt_ids = []
        prompt_ids = list(self.prompts.keys())
        for image_path in image_paths:
            prompt_id = int(image_path.split(os.path.sep)[-1].split("_")[-1].split(".")[0])
            if prompt_id not in prompt_ids:
                invalid_prompt_ids.append(prompt_id)

        if invalid_prompt_ids:
            raise ValueError(f"Invalid prompt ids : {invalid_prompt_ids}")

        return True

    async def _background_evaluate(self, image_paths: list[str], assigned_index: int) -> None:
        """Background evaluation that logs when complete."""
        try:
            preference_score, alignment_score, coherence_score = await self._evaluate_image_async(image_paths)
            await self.logger.log_at_index(assigned_index, 
                                           {f"rapidata/{self.model_name}_preference_image": preference_score, 
                                            f"rapidata/{self.model_name}_alignment_image": alignment_score, 
                                            f"rapidata/{self.model_name}_coherence_image": coherence_score,
                                            f"rapidata/{self.model_name}_average_score": (preference_score + alignment_score + coherence_score) / 3})
            print(f"Evaluation completed for step {assigned_index} with scores: {preference_score}, {alignment_score}, {coherence_score}")
        except Exception as e:
            print(f"Background evaluation failed for step {assigned_index}: {e}")
            import traceback
            traceback.print_exc()

    async def _evaluate_image_async(self, image_paths: list[str]) -> tuple[float, float, float]:
        """Async version of _evaluate_image method - runs all evaluations concurrently."""
        # Start all three evaluations concurrently and wait for completion
        preference_score, alignment_score, coherence_score = await asyncio.gather(
            self._evaluate_preference_image_async(image_paths),
            self._evaluate_alignment_image_async(image_paths),
            self._evaluate_coherence_image_async(image_paths)
        )

        return preference_score, alignment_score, coherence_score


    async def _evaluate_preference_image_async(self, image_paths: list[str]) -> float:
        """Async version that polls for results without blocking."""
        # Create and start the order - this is fast
        order = await asyncio.to_thread(
            lambda: self.client.order.create_compare_order(
                name=f"{self.model_name}_preference_image",
                instruction="Which image do you prefer?",
                datapoints=self._get_datapoints(image_paths)[0],
                validation_set_id="66d5ac99fc00255c2926df0c",
                responses_per_datapoint=IMAGE_EVALUATION_RESPONSES_PER_DATAPOINT,
            ).run()
        )

        # Wait for results asynchronously with polling using base class method
        results = await self._wait_for_results_async(order)
        results = results.to_pandas()
        average_score = results["A_summedUserScoresRatios"].mean()
        return float(average_score)
    
    async def _evaluate_alignment_image_async(self, image_paths: list[str]) -> float:
        """Async version that polls for results without blocking."""
        # Create and start the order - this is fast
        datapoints = self._get_datapoints(image_paths)
        order = await asyncio.to_thread(
            lambda: self.client.order.create_compare_order(
                name=f"{self.model_name}_alignment_image",
                instruction="Which image matches the description better?",
                datapoints=datapoints[0],
                contexts=datapoints[1],
                validation_set_id="6790c1b73711ca1ae1d948c3",
                responses_per_datapoint=IMAGE_EVALUATION_RESPONSES_PER_DATAPOINT,
            ).run()
        )
        # Wait for results asynchronously with polling using base class method
        results = await self._wait_for_results_async(order)
        results = results.to_pandas()
        average_score = results["A_summedUserScoresRatios"].mean()
        return float(average_score)
    
    async def _evaluate_coherence_image_async(self, image_paths: list[str]) -> float:
        """Async version that polls for results without blocking."""
        # Create and start the order - this is fast
        datapoints = self._get_datapoints(image_paths)
        order = await asyncio.to_thread(
            lambda: self.client.order.create_compare_order(
                name=f"{self.model_name}_coherence_image",
                instruction="Which image has more glitches and is more likely to be AI generated?",
                datapoints=datapoints[0],
                validation_set_id="67cafc95bc71604b08d8aa62",
                responses_per_datapoint=IMAGE_EVALUATION_RESPONSES_PER_DATAPOINT,
            ).run()
        )
        # Wait for results asynchronously with polling using base class method
        results = await self._wait_for_results_async(order)
        results = results.to_pandas()
        average_score = results["A_summedUserScoresRatios"].mean()
        return 1 - float(average_score) # Invert the score because the question is inverted

    def _get_datapoints(self, image_paths: list[str]) -> tuple[list[list[str]], list[str]]:
        if self.baseline_media is not None:
            return [[image_path, base_image_path] for image_path, base_image_path in zip(image_paths, self.baseline_media)], self.baseline_prompts
        
        base_image_path = "https://assets.rapidata.ai/4o-26-3-25"
        prompt_ids = [x.split(os.path.sep)[-1].split("_")[-1].split(".")[0] for x in image_paths]
        image_paths = [[image_path, f"{base_image_path}/{prompt_id}.webp"] for prompt_id, image_path in zip(prompt_ids, image_paths)]
        prompts = [self.prompts[int(prompt_id)] for prompt_id in prompt_ids]
        return image_paths, prompts
