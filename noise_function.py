import torch
import torch.nn.functional as F

def add_noise_to_tensor(image_tensor, noise_range=(-1, 1)):
    # Convert to floating-point tensor
    image_tensor = image_tensor.float()

    # Get the shape of the input tensor
    shape = image_tensor.shape

    # Generate random variables for each pixel within the specified range
    random_variables = torch.rand(shape, device=image_tensor.device) * (noise_range[1] - noise_range[0]) + noise_range[0]

    # Add noise to the image tensor element-wise
    noisy_image = image_tensor + random_variables * torch.std(image_tensor)

    return noisy_image