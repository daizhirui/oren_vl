import open_clip
from torch.nn import Module
from torchvision import transforms


class Extracter(Module):
    def __init__(self, clip_model_name="EVA02-L-14", clip_model_pretrained="merged2b_s4b_b131k", device="cuda"):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.clip_model_pretrained = clip_model_pretrained
        self.device = device

        self.model, self.train_transform, self.val_transform = open_clip.create_model_and_transforms(
            clip_model_name,
            clip_model_pretrained,
        )
        self.model.to(device)
        self.model.eval()

        self._model_forward_functions = {
            "EVA02-L-14": self._get_visual_features_eva,
        }
        self.model_forward_fn = self._model_forward_functions[clip_model_name]

        self._input_image_size = {
            "EVA02-L-14": (224, 224),
        }
        self._patch_size = {
            "EVA02-L-14": 14,
        }

    def get_transform_and_intrinsics(self, image_width, image_height, fx, fy, cx, cy):
        """
        Computes new camera intrinsics for the feature map size based on the original intrinsics and image dimensions.

        The function first calculates the scaling factors for width and height to fit the input image size required by
        the model. It then determines the overall scaling factor to maintain the aspect ratio. If the new dimensions
        exceed the input size, it calculates the necessary cropping and adjusts the principal point accordingly.
        Finally, it returns the new camera intrinsics (fx, fy, cx, cy) for the feature map size.

        Args:
            image_width (int): Original image width.
            image_height (int): Original image height.
            fx (float): Original focal length in x direction.
            fy (float): Original focal length in y direction.
            cx (float): Original principal point x coordinate.
            cy (float): Original principal point y coordinate.

        Returns:
            transform (torchvision.transforms.Compose): The transformation to apply to the input image.
            new_fx (float): New focal length in x direction for the feature map size.
            new_fy (float): New focal length in y direction for the feature map size.
            new_cx (float): New principal point x coordinate for the feature map size.
            new_cy (float): New principal point y coordinate for the feature map size.
        """
        input_width, input_height = self._input_image_size[self.clip_model_name]
        patch_size = self._patch_size[self.clip_model_name]

        scale_x = input_width / image_width
        scale_y = input_height / image_height

        scale = max(scale_x, scale_y)

        # resize the image to fit the input size while maintaining aspect ratio
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)

        assert (
            new_width >= input_width and new_height >= input_height
        ), "New dimensions must be at least as large as input size."

        rgb_transform = transforms.Compose(
            [
                transforms.Resize((new_height, new_width)),
                transforms.CenterCrop((input_height, input_width)),
            ]
        )
        depth_transform = transforms.Compose(
            [
                transforms.Resize(
                    (new_height, new_width),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=False,
                ),
                transforms.CenterCrop((input_height, input_width)),
                transforms.Resize(
                    (input_height // patch_size, input_width // patch_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=False,
                ),
            ]
        )

        # Intrinsics after resize + center crop (still in input-pixel coordinates).
        new_fx = fx * scale
        new_fy = fy * scale
        new_cx = cx * scale
        new_cy = cy * scale
        if new_width > input_width:
            new_cx -= (new_width - input_width) // 2
        elif new_height > input_height:
            new_cy -= (new_height - input_height) // 2

        # depth_transform downsamples by patch_size to feature resolution,
        # so rescale intrinsics into feature-pixel coordinates as well.
        new_fx /= patch_size
        new_fy /= patch_size
        new_cx /= patch_size
        new_cy /= patch_size

        return rgb_transform, depth_transform, new_fx, new_fy, new_cx, new_cy

    def _get_visual_features_eva(self, x):
        """
        Extracts normalized visual features from a EVA-based visual encoder.
        Returns features reshaped to (B, C, grid_size, grid_size).
        """
        x = self.val_transform(x.to(self.device))

        vision_model = self.model.visual.trunk  # EVA model

        x = vision_model.forward_intermediates(x, output_fmt="NCHW", norm=True, intermediates_only=True)[0]
        # x: features after the layer norm, shape (B, C, H, W)

        return x
