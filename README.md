
# FineTuning VisionLM on DGX Spark

 


## Overview
- Code and steps available to finetune VisionLM model on DGX Spark development environment. 
- Use of unsloth for finetuning.
- Uses multi-image use case. 
- Uses before and after satelite images of a disaster site to identfiy type and extent of damages 

## Installation
Finetuning Data: Data used for finetuning VLM model is XView2.
[XView2](https://xview2.org/)
Download this data in this folder structure:
<pre>
data
    \hold_images_labels_targets
        \hold
             \images
             \labels
             \targets
     \test_images_labels_targets
         \test
             \images
             \labels
             \targets
      \train_images_labels_targets
          \train
             \images
             \labels
             \targets
</pre>
Use this NVIDIA playbook to setup finetuning environment with Unsloth on DGX Spark:
[NVIDIA Playbook to run Unsloth on DGX Spark](https://build.nvidia.com/spark/unsloth/instructions)
Once in the docker environment, you can run the Jupyter notebook to run finetuning code.

## Code
Use 4 bit Qwen3-VL-8B-Instruct model for LORA Fineuning
```python
    model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
```

Setup finetuning params.
```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)
```

Prepare dataset
```python
from datasets import load_dataset
metadata = load_dataset("data/train_images_labels_targets/train/labels", split="train")
datasets = load_dataset("data/train_images_labels_targets/train/images", split="train")

def non_overlapping_pairs(seq):
    """Return non-overlapping consecutive pairs from `seq`.
    Works for general iterables (uses iterator pairing).
    Drops the last item if the length is odd."""
    it = iter(seq)
    return list(zip(it, it))

# Create pairs for dataset and metadata (non-overlapping).
dataset_pairs = non_overlapping_pairs(datasets)
metadata_pairs = non_overlapping_pairs(metadata)

def get_damage(items):
    """
    Select first item from items list based on priority order.
    
    Args:
        items: List of items ['damage', 'no-damage', 'destroyed']
        priority_list: Priority order ['destroyed', 'damage', 'no-damage']
    
    Returns:
        First item from items that matches priority order
    """
    priority_list = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']
    if len(items) == 0:
        return 'no-building'
    for priority_item in priority_list:
        if priority_item in items:
            return priority_item
    
    return None  # Return None if no match found



def get_conversation_response(post_data, pre_data, disaster):
    # Build simple list of type/condition dicts (one entry per feature)
    type_condition_list = [{'type': item['properties']['feature_type'], 'condition': item['properties']['subtype']} for item in post_data]
    my_list = [d["condition"] for d in type_condition_list]
    damage_type = get_damage(my_list)
    res = {'diaster_type': disaster, 'damage_type':damage_type }
    if damage_type is None:
        print('damage type not found')
    return res
```

Now the dataset is prepared, next step is Instruction Tuning to Convert data into Input (instruction/query) and Output (expected response) pairs. We use pre and post disaster statellite image pairs and response based on the damage information provided.
```python
instruction = "You are an expert satellite imagery analyst provided with pre and post disaster imageries. Describe accurately the extent of the damages."

def convert_to_conversation(dataset, metadata):
    disaster_type = metadata[0]['metadata']['disaster_type']
    response = get_conversation_response(metadata[0]['features']['lng_lat'],metadata[0]['features']['lng_lat'], disaster_type)

    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : dataset[1]['image']},
            {"type" : "image", "image" : dataset[0]['image']}]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : response} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(dataset, metadata) for (dataset, metadata) in zip(dataset_pairs, metadata_pairs)]

```

Lets check the inference response from the model before finetuning:

```python


FastVisionModel.for_inference(model) # Enable for inference!

image_1 = dataset_pairs[index][1]["image"]
image_2 = dataset_pairs[index][0]["image"]
#instruction = "You are an expert satellite imagery analyst who can identify extent of damages after a dsiaster. You are provided with 2 images, first is pre disaster imagery and second one is post disater imagery of the same location. Describe accurately the extent of the damages."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
print(input_text)
inputs = tokenizer(
    [image_1, image_2],
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")



from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
```

Here is the output:
```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>You are an expert satellite imagery analyst provided with pre and post disaster imageries. Describe accurately the extent of the damages.<|im_end|>
<|im_start|>assistant

Based on the provided satellite imagery, the analysis reveals a significant and widespread loss of forest cover in the second image, indicating a major destructive event such as a wildfire or deforestation. The changes are extensive and have occurred across large areas.

The key observations are:

- **Catastrophic Forest Loss**: There is a dramatic reduction in vegetation cover. In the first image, a dense, green forest dominates the landscape. In the second image, vast sections of this forest are now bare, brown, and charred earth, with only a few scattered, burnt-out trees remaining. The extent of this damage is immense, covering a significant portion of the area shown in the first image.

- **Transformation of Landscapes**: The area that was previously covered in lush green vegetation has been replaced by scorched, barren land. The color palette has shifted from vibrant greens to shades of brown and black, indicating severe burning and destruction.

- **Impact on Ecosystem**: The loss of the forest canopy is not just visual; it represents a major ecological disaster. This destruction will likely have long-term impacts on local wildlife, water cycles, and soil stability, as well as contributing to increased erosion and potential landslides.

In summary, the imagery indicates a severe and extensive damage event, with the forest cover being drastically reduced across the region, consistent with the aftermath of a major wildfire or large-scale deforestation. The transformation is both visually striking and ecologically significant.<|im_end|>

```

Obviously this response is good but is generic and too verbose for our use case. It is not confident on the type of disaster. We need to finetune the model so that it can detect the tye of disaster and the extent of the damage.

We have prepared instruction tuning data to do just that. here is an example record from the instruction tuning dataset:
```
{'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'You are an expert satellite imagery analyst provided with pre and post disaster imageries. Describe accurately the extent of the damages.'}, {'type': 'image', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1024x1024 at 0xED5C6A8CCA70>}, {'type': 'image', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1024x1024 at 0xED5C6A8CD4F0>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': {'diaster_type': 'flooding', 'damage_type': 'major'}}]}]}
```

Lets pass the dataset to the trainer to start finetuning:
```python
FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 100,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

trainer_stats = trainer.train()
```

After finetuning, we can perform inference on the finetuned model and validate the response:

```python
FastVisionModel.for_inference(model) # Enable for inference!
index = 777
image_1 = dataset_pairs[index][1]["image"]
image_2 = dataset_pairs[index][0]["image"]
#instruction = "You are an expert satellite imagery analyst. You are provided with 2 images for pre disaster imagery and post disater. Describe accurately the extent of the damages in json."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    [image_1, image_2],
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
print(convert_to_conversation(dataset_pairs[index], metadata_pairs[index]))
```

Lets check the response:
```
{'diaster_type': 'wind', 'damage_type': 'no-damage'}<|im_end|>
{'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'You are an expert satellite imagery analyst provided with pre and post disaster imageries. Describe accurately the extent of the damages.'}, {'type': 'image', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1024x1024 at 0xED5C6A9D5910>}, {'type': 'image', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1024x1024 at 0xED5C6A9D5A00>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': {'diaster_type': 'wind', 'damage_type': 'destroyed'}}]}]}

```
Much better. The response is exactly in the format we wanted.

We can run the inference on entire test data to measure the model performance metrics. First lets prepare data:
```python
metadata_test = load_dataset("data/test_images_labels_targets/test/labels", split='train')
datasets_test = load_dataset("data/test_images_labels_targets/test/images", split = 'train')
test_dataset_pairs = non_overlapping_pairs(datasets_test)
test_metadata_pairs = non_overlapping_pairs(metadata_test)
def get_ground_truth(dataset, metadata):
    disaster_type = metadata[0]['metadata']['disaster_type']
    return get_conversation_response(metadata[0]['features']['lng_lat'],metadata[0]['features']['lng_lat'], disaster_type)

ground_truth_data = [get_ground_truth(dataset, metadata) for (dataset, metadata) in zip(test_dataset_pairs, test_metadata_pairs)]
dis_type = [d['diaster_type']for d in ground_truth_data]
damage_type = [d['damage_type']for d in ground_truth_data]


```

```python
import traceback
import re
from tqdm import tqdm
progress_bar = tqdm(total=len(test_dataset_pairs))

FastVisionModel.for_inference(model) # Enable for inference!
def get_llm_response(image_pairs, index):
    progress_bar.update(1)
    image_1 = image_pairs[index][1]["image"]
    image_2 = image_pairs[index][0]["image"]
    #instruction = "You are an expert satellite imagery analyst. You are provided with 2 images for pre disaster imagery and post disater. Describe accurately the extent of the damages in json."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        [image_1, image_2],
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,temperature = 1.5, min_p = 0.1, pad_token_id=tokenizer.eos_token_id,
    )[0]

    assistant_response = tokenizer.decode(outputs[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    p = re.compile(r"(?<!\\)'") 
    valid_assistant_res = p.sub('"', assistant_response)
    #print(valid_assistant_res) 
    try:
        response_dict = json.loads(valid_assistant_res)
        return response_dict
        #print("Extracted Dictionary:", response_dict)
    except json.JSONDecodeError:
        #traceback.print_exc() 
        print("Failed to decode JSON. Raw output:", assistant_response)
    
llm_data = [get_llm_response(test_dataset_pairs, index) for index in range(len(test_dataset_pairs))]
dis_type_llm = [d['diaster_type']for d in llm_data]
damage_type_llm = [d['damage_type']for d in llm_data]
```

Calculate accuracy on test dataset
```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix
)
accuracy_score(damage_type, damage_type_llm)
```

I was able to acheive upto 70% accuracy on damage_type and 95% accuracy in disaster type.

### TODO
Try other models.
Try changing finetuning parameters.
Test with a different sets of datasets.
