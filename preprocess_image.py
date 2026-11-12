import numpy as np
from PIL import Image


class PreprocessImage:
    def __call__(self, path):
        """Provided: OpenCLIP shortest-edge bicubic resize and center crop."""
        with Image.open(path) as image:
            width, height = image.size
            if width <= height:
                size = (224, int(224 * height / width))
            else:
                size = (int(224 * width / height), 224)
            image = image.resize(size, Image.Resampling.BICUBIC)
            left = round((size[0] - 224) / 2)
            top = round((size[1] - 224) / 2)
            image = image.crop((left, top, left + 224, top + 224)).convert('RGB')
            return self.to_tensor(image)

    def to_tensor(self, image):
        """Provided: Convert the cropped PIL image to an OpenCLIP input array.
        Input: RGB PIL image of size (224,224). Output: float32 [1,3,224,224].
        Step 1: np.array(image,dtype=np.float32) gives [224,224,3]. Divide by 255.
        Step 2: Transpose axes (2,0,1), producing CHW [3,224,224].
        Step 3: Create float32 mean=[0.48145466,0.4578275,0.40821073] and
        std=[0.26862954,0.26130258,0.27577711], each reshaped to [3,1,1].
        Step 4: Compute (pixels-mean)/std channel by channel using broadcasting.
        Step 5: Add a leading batch axis; return the normalized batch array.
        These constants and the provided resize/crop must match both checkpoints.
        """
        pixels = np.array(image, dtype='float32').transpose(2, 0, 1) / np.float32(255)
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
        return ((pixels - mean) / std)[None]
