import numpy as np
from keras.models import load_model

def extract_patches(volume, patch_size=96, overlap=32):
    patches = []
    coords = []
    depth, height, width = volume.shape

    for z in range(0, depth - overlap, patch_size - overlap):
        for y in range(0, height - overlap, patch_size - overlap):
            for x in range(0, width - overlap, patch_size - overlap):
                z_end = z + patch_size
                y_end = y + patch_size
                x_end = x + patch_size

                if z_end > depth:
                    z_start = depth - patch_size
                    z_end = depth
                else:
                    z_start = z

                if y_end > height:
                    y_start = height - patch_size
                    y_end = height
                else:
                    y_start = y

                if x_end > width:
                    x_start = width - patch_size
                    x_end = width
                else:
                    x_start = x

                patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                patches.append(patch)
                coords.append((z_start, y_start, x_start))
                #print(patch.shape)
    return patches, coords

def create_gaussian_weight(patch_size, sigma):
    """Create a 3D Gaussian weight for blending patches."""
    center = patch_size // 2
    x = np.arange(0, patch_size) - center
    y = np.arange(0, patch_size) - center
    z = np.arange(0, patch_size) - center
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    gaussian_weight = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    return gaussian_weight

def rebuild_volume(patches, coords, volume_shape, patch_size=96, overlap=32, sigma=12, num_classes=3):
    depth, height, width = volume_shape
    volume = np.zeros((depth, height, width, num_classes), dtype=np.float32)
    weight_map = np.zeros((depth, height, width, num_classes), dtype=np.float32)
    
    gaussian_weight = create_gaussian_weight(patch_size, sigma)

    for patch, (z, y, x) in zip(patches, coords):
        # Calculate the end indices for each dimension
        z_end = min(z + patch_size, depth)
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)

        # Determine the actual size of the patch that fits within the volume bounds
        actual_patch_depth = z_end - z
        actual_patch_height = y_end - y
        actual_patch_width = x_end - x

        # Ensure patch dimensions match the volume slice dimensions
        patch_slice = patch[:actual_patch_depth, :actual_patch_height, :actual_patch_width, :]
        patch_weight = gaussian_weight[:actual_patch_depth, :actual_patch_height, :actual_patch_width, np.newaxis]

        volume[z:z_end, y:y_end, x:x_end, :] += patch_slice * patch_weight
        weight_map[z:z_end, y:y_end, x:x_end, :] += patch_weight

    # Avoid division by zero by setting zero values in the weight_map to one
    weight_map[weight_map == 0] = 1

    volume /= weight_map
    return volume


def process_and_predict_multi(volume, model='HRNET', patch_size=96, overlap=32, sigma=12):
    # Pad the volume if necessary
   
    model1 = load_model(model, compile = False)
    depth, height, width = volume.shape
    pad_depth = (patch_size - depth % patch_size) % patch_size
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    volume = np.pad(volume, ((0, pad_depth), (0, pad_height), (0, pad_width)), mode='reflect')
    volume_shape = volume.shape

    # Extract patches
    patches, coords = extract_patches(volume, patch_size, overlap)
    
    # Predict on each patch
    predicted_patches = []
    counter = 1
    for patch in patches:
        #print(patch.shape)
        #patch = np.squeeze(patch, axis=-1)  # Expand dimensions for model input
        #print(patch.shape)
        patch = np.stack((patch,)*3, axis=-1).astype(np.float32) / 255.0

    
        patch = np.expand_dims(patch, axis=0)
        #print(patch.shape)
        prediction = model1.predict(patch)
        predicted_patches.append(prediction.squeeze())
        print(f'Patch processed: {counter} of {len(patches)-1}')
        counter += 1
    # Rebuild the volume
    predicted_volume = rebuild_volume(predicted_patches, coords, volume_shape, patch_size, overlap, sigma)
    
    # Crop to original size
    predicted_volume = predicted_volume[:depth, :height, :width]
    
    # Threshold the predicted volume for binary segmentation
    predicted_volume = np.argmax(predicted_volume, axis=-1).astype(np.uint8)
    
    return predicted_volume

# # Example usage
# models_path = ['./3DVNet_VNet3D_fusev2_multi_noaug_-0.8259-73.keras','./HRNET_HRNET_multi_fusev2_dataset_noaug_-0.6808-23.keras', './resnext101_UNET_fusev2_dataset_noaug_-0.8191-97.keras', './SwinUnet_SwinUnet_fusev2_dataset_noaug_-0.7887-82.keras']

# for model in models_path:
#     input_volume = io.imread('/p/project1/claimd/Andre/Aimy/Dataset/cal_fusev2/test/Toray1202.tif')
#     predicted_volume = process_and_predict_multi(input_volume, model)
#     io.imsave(f'./39bb2_{os.path.basename(model)}.tif', predicted_volume.astype(np.uint8))
