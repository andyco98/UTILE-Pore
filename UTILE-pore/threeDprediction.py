import numpy as np
from skimage import io
from tensorflow.keras.models import load_model



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
    center = patch_size // 2
    ax = np.linspace(-(center), center, patch_size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel

def rebuild_volume(patches, coords, volume_shape, patch_size=96, overlap=32, sigma=12):
    depth, height, width = volume_shape
    volume = np.zeros(volume_shape, dtype=np.float32)
    weight_map = np.zeros(volume_shape, dtype=np.float32)
    
    gaussian_weight = create_gaussian_weight(patch_size, sigma)

    for patch, (z, y, x) in zip(patches, coords):
        z_end = min(z + patch_size, depth)
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)

        patch_weight = gaussian_weight[:z_end - z, :y_end - y, :x_end - x]

        volume[z:z_end, y:y_end, x:x_end] += patch * patch_weight
        weight_map[z:z_end, y:y_end, x:x_end] += patch_weight

    volume /= weight_map
    return volume

def process_and_predict(volume, model, patch_size=96, overlap=32, sigma=12):
    # Pad the volume if necessary
    model = load_model(model, compile = False)
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
        #print(patch.shape)
        ## Process input ##
        #process_input = sm.get_preprocessing('resnet34')
        #patch = process_input(patch)
        patch = np.expand_dims(patch, axis=0)
        #print(patch.shape)
        prediction = model.predict(patch)
        predicted_patches.append(prediction.squeeze())
        print(f'Patch processed: {counter} of {len(patches)-1}')
        counter += 1
    # Rebuild the volume
    predicted_volume = rebuild_volume(predicted_patches, coords, volume_shape, patch_size, overlap, sigma)
    
    # Crop to original size
    predicted_volume = predicted_volume[:depth, :height, :width]
    
    # Threshold the predicted volume for binary segmentation
    predicted_volume = (predicted_volume > 0.5).astype(np.uint8)
    
    return predicted_volume

# # Example usage
# model1 = '/p/project1/claimd/Andre/Aimy/3DVNet_VNet3D_Cal_OF_MPL_fuse_dataset_-0.8329-100.keras'  # Load your trained model here
# input_volume = io.imread('/p/project1/claimd/Andre/Aimy/Dataset/Sigracet28bc.tif')
# predicted_volume = process_and_predict(input_volume, model1)
# io.imsave('./Dataset/predicted_volume_Sigracet28bc_cal_fuse_aug_vnet.tif', predicted_volume.astype(np.uint8))
