# --------------------------------------------------------------------
# MultiModal Embodied Tracking
#  - VILA-1.5-3b-s2 model
# --------------------------------------------------------------------
import torch

from llava.constants import IMAGE_TOKEN_INDEX

from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class VILAModel:
    def __init__(
        self, 
        target_description: str,
        action_num: int = 4,
        model_path: str = "Efficient-Large-Model/VILA1.5-3b-s2",
        conv_mode: str = "vicuna_v1",
        temperature: float = 0.2,
    ) -> None:
        self.action_num = action_num
        self.model_path = model_path
        conv_mode = conv_mode
        self.temperature = temperature
        
        self.prompt = f"<image>]\nYou are a robot to track a target in the environment given the description: {target_description}. Your task is to select the most appropriate action based on the robot's current view. Here is the action list: \
            0. Stop. \
            1. Move forward. \
            2. Turn left. \
            3. Turn right. \
        You must follow the following rules: \
            1. Choose actions 1, 2, and 3 to track the target by obtaining a better view or getting closer to the target. \
            2. Choose action 0 only when you are close enough to the target object within about one meter. \
            3. If the target is not within the field of view, explore the environment first to find the target. \
            4. Try to keep the target in the center of the field of view. \
            5. Select the action by only output the number with the action. For example: If you want to choose \"Stop\", just output \"0\". \
            6. Your output must only be a single integer from 0 to 3. \
            7. Never explain your choice. \
            8. Never include information in your answer that is not relevant to the question. \
            9. Robot's current view is the given image."
        
        self.init_model()
        
    def init_model(self):
        disable_torch_init()

        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.model_path, model_name, model_base=None)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], self.prompt)
        conv.append_message(conv.roles[1], None)
        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [self.stop_str]

        prompt = conv.get_prompt()
        self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        self.stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

    def predict(self, image, depth=None):
        """
        Get the action based on the action_prob
            stop = 0
            move_forward = 1
            turn_left = 2
            turn_right = 3
        """
        images_tensor = process_images(image, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                self.input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[self.stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(self.stop_str):
            outputs = outputs[: -len(self.stop_str)]
        outputs = outputs.strip()
        
        return outputs