# Crowd Evaluation for T2I

A Python library for integrating crowd evaluation into your machine learning training loops. This library provides asynchronous, non-blocking evaluation of model outputs (currently supporting image generation) with automatic logging to Weights & Biases (wandb).

## Features

- **Asynchronous Evaluation**: Evaluations run in the background without blocking your training loop
- **Wandb Integration**: Results are automatically logged to your wandb runs with proper ordering
- **Image Evaluation**: Built-in support for evaluating generated images on multiple criteria
- **Crowd-in-the-Loop**: Uses [Rapidata](https://rapidata.ai/) for high-quality crowd evaluation
- **Easy Integration**: Add evaluation to your training loop with just a few lines of code

## Quick Start

```python
import wandb
from src.crowd_eval.checkpoint_evaluation.image_checkpoint_evaluator import ImageEvaluator

# Initialize wandb
run = wandb.init(project="my-project")

# Create evaluator
evaluator = ImageEvaluator(wandb_run=run, model_name="my-model")

# In your training loop
for step in range(100):
    # ... your training code ...
    
    # Generate or load validation images (every N steps)
    if step % 10 == 0:
        validation_images = ["path/to/image_1.png", "path/to/image_2.png"]
        
        # Fire-and-forget evaluation - returns immediately!
        evaluator.evaluate(validation_images)
    
    # ... continue training ...

# Wait for all evaluations to complete before finishing
evaluator.wait_for_all_evaluations()
run.finish()
```

## Installation

### Prerequisites

- Python 3.9+
- A [Rapidata](https://rapidata.ai/) account with API credentials
- A [Weights & Biases](https://wandb.ai/) account

### Pip install (recommended)

```bash
pip install crowd-eval
```

### Local install

#### Prerequisites
Install uv if you haven't already:
```bash
# For MacOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# For Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Setup Instructions

1. Create and activate a virtual environment:
    ```bash
    uv venv

    # On Unix/macOS
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\activate
    ```
2. Install dependencies:
    ```bash
    uv sync
    ```

### Environment Setup (optional for different usecases)

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key  # If running the example file
RAPIDATA_CLIENT_ID=your_rapidata_client_id # If running on a server
RAPIDATA_CLIENT_SECRET=your_rapidata_client_secret # If running on a server
```

## Detailed Usage

### Image Evaluation

The `ImageEvaluator` evaluates generated images on three key metrics:

1. **Preference**: Overall crowd preference for the image
2. **Alignment**: How well the image matches its text description
3. **Coherence**: Visual quality and absence of artifacts

### Image Requirements

For the evaluator to function properly, your image files should adhere to the following naming convention: the image name must end with "_{prompt_id}". The rest of the filename structure is not significant.

Where `{prompt_id}` corresponds to prompt IDs from the evaluation dataset. The evaluator will automatically validate that your images match available prompts.

### Complete Example with Image Generation

#### To run this, make sure you run the following commands:
```bash
uv venv
source .venv/bin/activate
uv sync
uv add openai dotenv
```

and log in to wandb:
```bash
wandb login
```

```python
import os
import sys
import openai
import requests
import wandb
from src.crowd_eval.checkpoint_evaluation.image_checkpoint_evaluator import ImageEvaluator
from dotenv import load_dotenv

load_dotenv()

# Setup
openai.api_key = os.getenv("OPENAI_API_KEY")
run = wandb.init(project="dalle-evaluation")
evaluator = ImageEvaluator(wandb_run=run, model_name="dalle-3")

def generate_and_save_image(prompt: str, file_location: str) -> str:
    """Generate image using DALL-E and save to disk."""
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    
    # Download and save image
    image_url = response.data[0].url
    image_data = requests.get(image_url).content
    with open(file_location, 'wb') as f:
        f.write(image_data)
    
    return file_location

if __name__ == "__main__":
    # Training simulation
    for step in range(3):
        # Simulate training
        run.log({"Some training metric": step})
        
        # Generate images for evaluation (using first 2 prompts)
        validation_images = [
            generate_and_save_image(prompt, f"validation_images/generated_image_run_{step}_{id}.png")
            for id, prompt in list(evaluator.prompts.items())[:2]
        ]
        
        # Evaluate asynchronously
        evaluator.evaluate(validation_images)

    print("This will run immediately, but the evaluations will run in the background.")

    # Wait for all evaluations
    evaluator.wait_for_all_evaluations()
    run.finish()
```

## Custom Baseline

By default, the `ImageEvaluator` compares your generated images against a pre-defined set of baseline images from GPT-4o. However, you can define your own custom baseline images and prompts for more targeted evaluation scenarios.

### Setting Up a Custom Baseline

Use the `define_baseline()` method to specify your own baseline images and prompts:

```python
# Define custom baseline with your own images and prompts
evaluator.define_baseline(
    image_paths=[
        "path/to/baseline_image_1.png",
        "path/to/baseline_image_2.png",
        "https://example.com/remote_baseline.jpg"  # URLs also supported
    ],
    prompts=[
        "A serene mountain landscape",
        "A futuristic city skyline", 
        "An abstract geometric pattern"
    ]
)
```

### How Custom Baselines Work

When you define a custom baseline:

1. **Image Naming**: Your generated images no longer need to follow the `*_{prompt_id}.png` naming convention
2. **Direct Comparison**: Each generated image is compared directly against the corresponding baseline image at the same index
3. **Custom Prompts**: The evaluation uses your provided prompts instead of the default dataset
4. **Matched Pairs**: The number of generated images must match the number of baseline images

### Complete Example with Custom Baseline

```python
import wandb
from src.crowd_eval.checkpoint_evaluation.image_checkpoint_evaluator import ImageEvaluator

# Initialize
run = wandb.init(project="custom-baseline-eval")
evaluator = ImageEvaluator(wandb_run=run, model_name="my-model")

# Set up custom baseline
evaluator.define_baseline(
    image_paths=[
        "baselines/reference_1.png",
        "baselines/reference_2.png"
    ],
    prompts=[
        "A red sports car",
        "A sunset over the ocean"
    ]
)

# Training loop
for step in range(10):
    # Your training code here...
    
    if step % 5 == 0:
        # Generate images for your custom prompts
        generated_images = [
            f"outputs/step_{step}_car.png",      # Compares against baselines/reference_1.png
            f"outputs/step_{step}_sunset.png"   # Compares against baselines/reference_2.png
        ]
        
        # Evaluate against your custom baseline
        evaluator.evaluate(generated_images)

# Wait for evaluations and finish
evaluator.wait_for_all_evaluations()
run.finish()
```

### Benefits of Custom Baselines

- **Domain-Specific Evaluation**: Use baselines relevant to your specific use case
- **Consistent Comparison**: Compare against the same reference images across training runs  
- **Flexible Prompts**: Use any prompts that make sense for your model's intended application
- **Quality Control**: Establish known-good reference images as quality benchmarks




## Troubleshooting

### Common Issues

**"Invalid prompt ids" error:**
- Ensure image filenames follow the pattern: `*_{prompt_id}.png`
- Check that `{prompt_id}` exists in the evaluation dataset

**Evaluations not appearing in wandb:**
- Call `evaluator.wait_for_all_evaluations()` before `run.finish()`
- Check your Rapidata API credentials
- Verify internet connectivity for API calls

**"Module not found" error:**
- Ensure you have the correct dependencies installed
- Ensure your example code is run from the root of the repository

### Environment Variables

Required:
- `RAPIDATA_CLIENT_ID`: Your Rapidata client ID (Not required if running locally)
- `RAPIDATA_CLIENT_SECRET`: Your Rapidata client secret (Not required if running locally)

Optional:
- `OPENAI_API_KEY`: For image generation examples
