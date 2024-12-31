from cog import BasePredictor, Input, Path
import os, random, sys
import numpy as np
import torch
from PIL import Image
from typing import List
from openai import OpenAI

from huggingface_hub import snapshot_download
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from segment_anything import SamPredictor, build_sam, SamAutomaticMaskGenerator

from app.src.vlm_pipeline import (
    vlm_response_editing_type,
    vlm_response_object_wait_for_edit,
    vlm_response_mask,
    vlm_response_prompt_after_apply_instruction
)
from app.src.brushedit_all_in_one_pipeline import BrushEdit_Pipeline
from app.utils.utils import load_grounding_dino_model
from dotenv import load_dotenv

load_dotenv()

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Initialize device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16

        # Download and set up model paths
        print("[LOG] Setting up model paths...")
        self.brushedit_path = "models/"
        if not os.path.exists(self.brushedit_path):
            self.brushedit_path = snapshot_download(
                repo_id="TencentARC/BrushEdit",
                local_dir=self.brushedit_path,
                token=os.getenv("HF_TOKEN"),
            )

        # Initialize paths
        self.base_model_path = os.path.join(self.brushedit_path, "base_model/realisticVisionV60B1_v51VAE")
        self.brushnet_path = os.path.join(self.brushedit_path, "brushnetX")
        self.sam_path = os.path.join(self.brushedit_path, "sam/sam_vit_h_4b8939.pth")
        self.groundingdino_path = os.path.join(self.brushedit_path, "grounding_dino/groundingdino_swint_ogc.pth")

        # Load BrushNet
        print("[LOG] Loading BrushNet...")
        self.brushnet = BrushNetModel.from_pretrained(
            self.brushnet_path,
            torch_dtype=self.torch_dtype
        )
        
        # Initialize pipeline
        self.pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            self.base_model_path,
            brushnet=self.brushnet,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=False
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        # Load SAM
        print("[LOG] Loading SAM...")
        self.sam = build_sam(checkpoint=self.sam_path)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        self.sam_automask_generator = SamAutomaticMaskGenerator(self.sam)

        # Load GroundingDINO
        print("[LOG] Loading GroundingDINO...")
        config_file = 'app/utils/GroundingDINO_SwinT_OGC.py'
        self.groundingdino_model = load_grounding_dino_model(
            config_file,
            self.groundingdino_path,
            device=self.device
        )

        # Initialize GPT-4o
        print("[LOG] Initializing GPT-4o...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.vlm_model = OpenAI(api_key=api_key)
        self.vlm_processor = ""  # GPT-4o doesn't need a processor

    def predict(
        self,
        image: str,
        prompt: str,
        target_prompt: str = None,
        mask: str = None,
        negative_prompt: str = "ugly, low quality",
        control_strength: float = 1.0,
        seed: int = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        num_samples: int = 4,
        blending: bool = True,
        category: str = None,
    ) -> List[str]:
        """Run inference using the BrushEdit pipeline"""

        # Load and preprocess input image
        max_image_size = 640
        input_image = Image.open(image).convert("RGB")

        # Get original dimensions
        original_width, original_height = input_image.size

        # Calculate aspect ratio
        aspect_ratio = original_width / original_height

        # Determine new dimensions while keeping the aspect ratio
        if original_width > original_height:
            new_width = max_image_size
            new_height = int(max_image_size / aspect_ratio)
        else:
            new_height = max_image_size
            new_width = int(max_image_size * aspect_ratio)

        # Resize the image with new dimensions
        input_image = input_image.resize((new_width, new_height), Image.NEAREST)

        # Convert to numpy array
        original_image = np.array(input_image)

        # Process or generate mask
        if mask:
            original_mask = np.array(Image.open(mask).convert("L"))
        else:
            # Get editing category if not provided
            if not category:
                category = vlm_response_editing_type(
                    self.vlm_processor,
                    self.vlm_model,
                    original_image,
                    prompt,
                    self.device
                )

            # Get object to edit
            object_wait_for_edit = vlm_response_object_wait_for_edit(
                self.vlm_processor,
                self.vlm_model,
                original_image,
                category,
                prompt,
                self.device
            )

            # Generate mask
            original_mask = vlm_response_mask(
                self.vlm_processor,
                self.vlm_model,
                category,
                original_image,
                prompt,
                object_wait_for_edit,
                self.sam,
                self.sam_predictor,
                self.sam_automask_generator,
                self.groundingdino_model,
                self.device
            )

        # Ensure mask is in correct format
        if original_mask.ndim == 2:
            original_mask = original_mask[:, :, None]
        original_mask = np.clip(original_mask, 0, 255).astype(np.uint8)

        # Save the original mask
        mask_output_path = "./output/original_mask.png"
        Image.fromarray(original_mask.squeeze()).save(mask_output_path)

        # Get target prompt if not provided
        if not target_prompt:
            target_prompt = vlm_response_prompt_after_apply_instruction(
                self.vlm_processor,
                self.vlm_model,
                original_image,
                prompt,
                self.device
            )

        print(f"target_prompt: {target_prompt}")
        # Set up generator
        if seed is None:
            seed = random.randint(0, 2147483647)
        generator = torch.Generator(self.device).manual_seed(seed)

        # Run BrushEdit pipeline
        with torch.autocast(self.device):
            images, mask_image, mask_np, init_image_np = BrushEdit_Pipeline(
                self.pipe,
                target_prompt,
                original_mask,
                original_image,
                generator,
                num_inference_steps,
                guidance_scale,
                control_strength,
                negative_prompt,
                num_samples,
                blending
            )

        # Save outputs
        output_paths = []
        for i, img in enumerate(images):
            output_path = f"./output/output_{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

    def resize(
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """
        Crops and resizes an image while preserving the aspect ratio.

        Args:
            image (Image.Image): Input PIL image to be cropped and resized.
            target_width (int): Target width of the output image.
            target_height (int): Target height of the output image.

        Returns:
            Image.Image: Cropped and resized image.
        """
        # Original dimensions
        resized_image = image.resize((target_width, target_height), Image.NEAREST)
        return resized_image

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    output = predictor.predict(
        image="assets/spider_man_rm/spider_man.png",
        prompt="remove the Christmas hat."
    )
    
