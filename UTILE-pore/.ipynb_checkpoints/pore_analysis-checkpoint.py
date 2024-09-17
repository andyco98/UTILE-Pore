import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import csv
from skimage.measure import marching_cubes
from skimage import measure
import cv2
from scipy.stats import variation
from scipy.ndimage import binary_dilation
from visualization import *
import openpnm as op

def crop_3d_array(array, target_shape):
    """
    Crops a 3D numpy array to the target shape.

    Parameters:
    - array: The original 3D numpy array.
    - target_shape: A tuple representing the desired shape (depth, height, width).

    Returns:
    - cropped_array: The cropped 3D numpy array.
    """
    original_shape = array.shape
    print(original_shape)
    
    # Ensure the target shape is smaller than or equal to the original shape
    assert all(t <= o for t, o in zip(target_shape, original_shape)), "Target shape must be smaller than or equal to the original shape."
    
    # Calculate the start indices for cropping
    start_indices = [(o - t) // 2 for o, t in zip(original_shape, target_shape)]
    
    # Calculate the end indices for cropping
    end_indices = [start + t for start, t in zip(start_indices, target_shape)]
    
    # Perform cropping
    cropped_array = array[start_indices[0]:end_indices[0],
                          start_indices[1]:end_indices[1],
                          start_indices[2]:end_indices[2]]
    
    return cropped_array

def calculate_surface_roughness_from_surface(binary_volume, voxel_size = 5, deep = 10, cap_thickness = True):
    """
    Calculate the arithmetic mean roughness (Ra) and root mean square roughness (Rq) of a surface.
    
    Parameters:
    - verts: Vertices of the surface mesh.
    
    Returns:
    - Ra: Arithmetic mean roughness.
    - Rq: Root mean square roughness.
    """
    if cap_thickness:
        try:
            # Limit the binary volume to the first `num_slices`
            sliced_volume = binary_volume[:deep, :, :]
        except:
            sliced_volume = binary_volume[:, :]
    else: sliced_volume = binary_volume
        # Extract 
    verts, faces, _, _ = marching_cubes(sliced_volume, level=0)

    # Extract the height values (Y-axis assuming depth, height, width ordering)
    z_values = verts[:, 0]

    # Adjust the heights relative to the minimum value to focus on local variation
    adjusted_z_values = z_values - np.min(z_values)
    print(adjusted_z_values)
    # Calculate roughness in voxel units
    mean_z = np.mean(adjusted_z_values)
    Ra = np.mean(np.abs(adjusted_z_values - mean_z))
    Rq = np.sqrt(np.mean((adjusted_z_values - mean_z) ** 2))
    
    # Convert roughness from voxel units to physical units using the voxel size
    Ra = Ra * voxel_size
    Rq = Rq * voxel_size
    
    return Ra, Rq

def save_results_to_csv(results, porosity, avg_pore, sd, Ra, Rq, permeability, tortuosity,  filepath):
    """
    Saves the pore size distribution results and porosity to a CSV file.
    
    Parameters:
    - results: A dictionary containing the pore size distribution results.
    - porosity: The calculated porosity of the volume.
    - filepath: The path to the CSV file where the results will be saved.
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['Porosity', porosity])
        writer.writerow(['Average_Pore_Size', avg_pore])
        writer.writerow(['Standrad_Deviation', sd])
        writer.writerow(['Tortuosity', tortuosity])
        # Write header for pore size distribution
        
        #Write surface roughness values
        writer.writerow(['Arithmetic_Mean_Roughness', Ra])
        writer.writerow(['Root_Square_Mean_Roughness', Rq])
        writer.writerow(['Permeability', permeability])
        writer.writerow(['Bin_Centers', 'PDF'])

        for bin_center, pdf_value in zip(results['bin_centers'], results['pdf']):
            writer.writerow([bin_center, pdf_value])

def open_tiff_stack(filepath):
    with tiff.TiffFile(filepath) as tif:
        images = tif.asarray()
        print(images.shape)
    return images

def calculate_psd(filepath, csv_file, case_name, voxel_size=5):  # Added voxel_size parameter with a default of 5 microns

    # Ensure the image is binary
    binary_image_3d = np.where(image_3d == 0, 1, 0)
    porosity = ps.metrics.porosity(binary_image_3d)
    print(f"Porosity: {porosity}")

    sizes = ps.filters.porosimetry(im=binary_image_3d)
    print(f"Sizes shape: {sizes.shape}")

    results = ps.metrics.pore_size_distribution(sizes)
    print("Results:", results)

    bin_centers = results['bin_centers']
    pdf = results['pdf']
    
    norm_pdf = pdf/ np.sum(pdf)
    # Calculate average and standard deviation using the PDF and bin centers
    average_pore_size = np.sum(bin_centers * norm_pdf) * voxel_size  # Scale to microns
    variance = np.sum((bin_centers * voxel_size - average_pore_size)**2 * pdf)  # Scale variance to microns^2
    std_deviation = np.sqrt(variance)

    print(f"Average Pore Size (microns): {average_pore_size}")
    print(f"Standard Deviation (microns): {std_deviation}")

    # Plot the pore size distribution
    fig, ax = plt.subplots()
    ax.plot(bin_centers * voxel_size, pdf, 'bo-', label=f'Pore Size Distribution\nAvg: {average_pore_size:.2f} µm\nSD: {std_deviation:.2f} µm')  # Scale x-axis to microns
    ax.set_xlabel('Pore radius (microns)')
    ax.set_ylabel('Frequency')
    ax.set_title('Pore Size Distribution in 3D')
    ax.legend()
    plt.savefig(f'./{case_name}/psd_plot.png')
    plt.close(fig)  # Close the figure to avoid display issues in scripts

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### Pore Size Distribution Analysis #####'])
        writer.writerow(['Porosity', porosity])
        writer.writerow(['Average_Pore_Size', average_pore_size])
        writer.writerow(['Standrad_Deviation', std_deviation])

    return porosity, results, average_pore_size, std_deviation

def calculate_ssa(binary_volume, voxel_size=5):
    """
    Calculates the specific surface area (SSA) from a 3D binary segmented volume.
    
    Parameters:
    - binary_volume: A 3D numpy array (binary) where 1 represents the material phase and 0 represents the void phase.
    - voxel_size: The physical size of each voxel, if known, in units such as microns (default is 1).
    
    Returns:
    - ssa: The specific surface area, in units of surface area per volume.
    """
    # Ensure the binary volume is in the correct format
    binary_volume = np.asarray(binary_volume).astype(bool)
    verts, faces, n, v = marching_cubes(binary_volume, level=0, spacing=(voxel_size, voxel_size, voxel_size))
    # Surface area estimation using marching cubes algorithm from scikit-image
    #verts, faces, _, _ = marching_cubes(binary_volume, level=0, spacing=(voxel_size, voxel_size, voxel_size))
    
    #mesh = ps.tools.mesh_region(verts = verts, faces = faces)
    surface_area = ps.metrics.mesh_surface_area(verts = verts, faces = faces)

    # Calculate volume occupied by the solid phase (1s in the binary volume)
    solid_volume = np.sum(binary_volume) * (voxel_size ** 3)

    # Calculate specific surface area (SSA) as surface area divided by volume
    ssa = surface_area / solid_volume
    return ssa

def estimate_permeability(porosity, csv_file, specific_surface_area, tortuosity=1.5):
    """
    Estimate permeability using the Kozeny-Carman equation.
    
    Parameters:
    - porosity: The porosity of the material.
    - specific_surface_area: The specific surface area of the material.
    - tortuosity: The tortuosity of the porous structure (default is 1.5).
    
    Returns:
    - permeability: The estimated permeability.
    """
    # Kozeny-Carman constant, typically around 5 for packed spheres
    k = 5
    
    # Kozeny-Carman equation for permeability
    permeability = (porosity**3) / (k * tortuosity**2 * (1 - porosity)**2 * specific_surface_area**2)
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### Estimated Permeability Kozeny-Carman #####'])
        writer.writerow(['Permeability', permeability])
    return permeability

def calculate_solid_surface_ratio(binary_image_3d,csv_file, side,gdl=1):
    binary_image_3d = np.where(binary_image_3d==gdl, 1, 0)
    if side == 'top':
        first_layer = binary_image_3d[0, :, :]
    else:
        first_layer = binary_image_3d[-1, :, :]  # Bottom slice
    
    h,w = first_layer.shape
    print(first_layer.shape)
    total_px = h*w

    white_px = np.sum(first_layer == 1)
    solid_ratio = white_px / total_px
    print('Solid surface ratio:', solid_ratio)

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### Solid surface ratio #####'])
        writer.writerow(['Solid_surface_ratio', solid_ratio])
    return solid_ratio

def MPL_GDL_thickness(volume, csv_file, axis=0, mpl=2, gdl=1, voxel_size=5):
    if len(np.unique(volume)) == 2:
        thickness = volume[0]
        print('GDL thickness: ', thickness)
        return thickness
    else:
        # Find the positions of the mpl in the 3D volume
        mpl_mask = binary_image_3d == mpl
        #print(mpl_mask.shape)
        
        # Sum along the height and width (axes 1 and 2) to get the number of MPL voxels per slice (along the Z-axis)
        thickness_along_z = np.sum(mpl_mask, axis=(1, 2))
        #print('thickness_along_z (MPL) ', thickness_along_z)
        
        # Non-zero thickness is the region where the layer is present
        non_zero_slices = np.where(thickness_along_z > 0)[0]
        #print('non zero layer (MPL)', non_zero_slices.shape)
        
        if len(non_zero_slices) == 0:
            raise ValueError("The specified MPL class is not present in the volume.")
        
        mpl_volume = np.where(binary_image_3d==2, 1,0)
        mpl_thickness = np.sum(mpl_volume, axis=0)  # Sum along the Z-axis
        mpl_avg_thickness = int(np.mean(mpl_thickness))
        print('MPL avg Thickness',mpl_avg_thickness)       

        gdl_volume = np.where(binary_image_3d==1, 1,0)
        gdl_thickness = np.sum(gdl_volume, axis=0)  # Sum along the Z-axis
        gdl_avg_thickness = int(np.mean(gdl_thickness))
        print('GDL avg Thickness',gdl_avg_thickness)

        # Calculate MPL thickness as the number of slices where the layer is present
        mpl_thickness = (non_zero_slices[-1] - non_zero_slices[0] + 1)* voxel_size

        # Check GDL layer thickness
        gdl_mask = binary_image_3d == gdl
        
        # Sum along the height and width (axes 1 and 2) to get the number of GDL voxels per slice (along the Z-axis)
        thickness_along_z = np.sum(gdl_mask, axis=(1, 2))
        print('thickness_along_z (GDL) ', thickness_along_z)
        
        # Non-zero thickness is the region where the layer is present
        non_zero_slices = np.where(thickness_along_z > 0)[0]
        
        if len(non_zero_slices) == 0:
            raise ValueError("The specified GDL class is not present in the volume.")
        
        # Calculate GDL thickness as the number of slices where the layer is present
        max_gdl_thickness = (non_zero_slices[-1] - non_zero_slices[0] + 1)* voxel_size
        print('Max GDL Thickness', max_gdl_thickness, 'Max MPL Thickness', mpl_thickness)
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write porosity
            writer.writerow(['##### MPL/GDL Thicknesses #####'])
            writer.writerow(['GDL_thickness', max_gdl_thickness])
            writer.writerow(['MPL_thickness', mpl_thickness])
        return mpl_thickness, gdl_thickness

def MPL_crack_analysis(mpl_layer, case_name, csv_file, mpl=2, slice_idx=0, from_top=True, voxel_size=5):
    """
    Analyzes cracks in a single slice (top or bottom) of the MPL layer and plots the crack size distribution.
    
    Parameters:
    - mpl_layer: Binary mask of the MPL layer (3D)
    - slice_idx: The index of the slice to analyze (default is 0 for the first slice)
    - from_top: Whether to analyze from the top (True) or bottom (False)
    
    Returns:
    - crack_ratio: Ratio of the crack area to the total area in the slice
    - crack_count: Number of cracks found in the slice
    - crack_labels: Labeled image where each crack has a unique label
    - crack_sizes: List of sizes (area in pixels) of the individual cracks
    - slice_image: The analyzed slice image
    """
    # Select the slice from the top or bottom
    if from_top:
        slice_image = mpl_layer[slice_idx, :, :]
    else:
        slice_image = mpl_layer[-(slice_idx+1), :, :]  # Bottom slice

    # Binarize the slice image (1 for MPL, 0 for cracks)
    slice_image = np.where(slice_image == mpl, 1, 0)
    
    # Crack ratio calculation
    h, w = slice_image.shape
    total_px = h * w
    white_px = np.sum(slice_image == 1)
    crack_ratio = 1 - (white_px / total_px)  # Ratio of crack area to total area

    # Analyze cracks in the selected slice (0 corresponds to cracks/pore spaces)
    crack_labels = measure.label(slice_image == 0, connectivity=2)  # 2D connectivity
    crack_count = np.max(crack_labels)  # Number of cracks
    
    # Crack size distribution analysis
    crack_sizes = []
    regions = measure.regionprops(crack_labels)
    for region in regions:
        if region.area > 10:
            crack_sizes.append(region.area)  # Append the area of each crack
    
    crack_sizes = crack_sizes * (voxel_size**2)
    # Define the number of bins and the range in logarithmic scale
    min_bin = np.log10(min(crack_sizes))  # Minimum value for log scale
    max_bin = np.log10(max(crack_sizes))  # Maximum value for log scale
    bins = np.logspace(min_bin, max_bin, num=20)  # Create 20 bins spaced logarithmically
    
    # Plot the crack size distribution (PSD)
    plt.figure()
    plt.hist(crack_sizes, bins=bins, edgecolor='black')
    plt.xscale('log')
    plt.title("Crack Size Distribution")
    plt.xlabel("Crack Area (microns^2)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'./{case_name}/crack_size_distirbution.png')
    plt.close()

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### Crack Analysis #####'])
        writer.writerow(['Crack_ratio', crack_ratio])
        writer.writerow(['Crack_count', crack_count])   
    return crack_ratio, crack_count, crack_labels, crack_sizes, slice_image

def plot_crack_labels(slice_image, crack_labels):
    """
    Plots the crack labels on the slice for visualization.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(slice_image, cmap='gray')
    ax[0].set_title("MPL Layer Slice")
    
    ax[1].imshow(crack_labels, cmap='nipy_spectral')
    ax[1].set_title("Cracks Labeled")
    
    plt.savefig(f'./{case_name}/mpl_cracks_map.png')
    plt.close(fig)

def layer_pore_size_distribution(volume, gdl, visualize=False,voxel_size=5):
    mean_values = []
    sd_values = []
    pore_count_values = []

    # Convert the GDL volume into binary (pore = 1, solid = 0)
    binary_volume = np.where(volume == 0, 1, 0)

    for i in range(0, binary_volume.shape[0] - 1):

        white_pore_slice = binary_volume[i, :, :]
        white_pore_slice = np.where(white_pore_slice == 0, 1, 0)
        white_pore_slice = white_pore_slice.astype(np.uint8)

        # Find contours in the slice
        contours, _ = cv2.findContours(white_pore_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store the sizes (areas) of each pore in this slice
        pore_sizes = []

        for contour in contours:
            # Calculate the area of the contour (pore size in pixels)
            area = cv2.contourArea(contour)

            # Convert area from pixels to square microns (real-world area)
            real_area = area * (voxel_size ** 2)  # Area in square microns
            pore_sizes.append(real_area)

        pore_count = len(pore_sizes)

        # Only append values if we have detected pores
        if pore_sizes:
            mean_values.append(np.mean(pore_sizes))
            sd_values.append(np.std(pore_sizes))
        else:
            mean_values.append(0)
            sd_values.append(0)

        pore_count_values.append(pore_count)
                # Visualization: Display the current slice with contours
        if visualize:
            # Create a color image to display contours in color
            color_slice = cv2.cvtColor(white_pore_slice * 255, cv2.COLOR_GRAY2BGR)

            # Draw contours on the image
            cv2.drawContours(color_slice, contours, -1, (0, 255, 0), 1)  # Green contours

            # Plot the slice with the contours
            plt.figure(figsize=(6, 6))
            plt.imshow(color_slice)
            plt.title(f"Slice {i} with Contours")
            plt.axis('off')  # Turn off the axis
            plt.show()

    # Calculate global mean and standard deviation
    mean_global = np.mean(mean_values)
    sd_global = np.mean(sd_values)
    print('Pore count:', pore_count)
    print('Mean values per slice:', mean_values)
    print('Standard deviation per slice:', sd_values)
    print('Global mean pore size (in square microns):', mean_global)
    print('Global standard deviation of pore size (in square microns):', sd_global) 

def MPL_heatmap(volume,case_name, mpl):
    # First lets do a heatmap wit the MPL surface
    mpl_volume = np.where(volume == mpl, 1, 0)

    xy_density = np.sum(mpl_volume, axis=0)

    fig, ax =  plt.subplots()
    ax.imshow(xy_density, cmap='jet')
    ax.set_title('XY Plane Density')
    ax.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False)
    plt.savefig(f'./{case_name}/mpl_heatmap.png')
    plt.close(fig)

def calculate_local_roughness(verts, voxel_size=5, region_size=10):
    """
    Calculate local roughness values (Ra) across small regions of the MPL surface.
    
    Parameters:
    - verts: Vertices of the surface mesh.
    - voxel_size: The physical size of each voxel, in units such as microns.
    - region_size: The size of each region for local roughness calculation (in number of vertices).
    
    Returns:
    - local_Ra_values: A list of local Ra values for each region.
    """
    local_Ra_values = []
    num_verts = verts.shape[0]
    
    for i in range(0, num_verts, region_size):
        region_verts = verts[i:i + region_size]
        if len(region_verts) == 0:
            continue
        
        # Extract the height values for the region
        z_values = region_verts[:, 0]
        adjusted_z_values = z_values - np.min(z_values)
        
        # Calculate the local Ra (arithmetic mean roughness) for the region
        mean_z = np.mean(adjusted_z_values)
        Ra = np.mean(np.abs(adjusted_z_values - mean_z))
        Ra = Ra * voxel_size  # Scale to physical units
        local_Ra_values.append(Ra)
    
    return local_Ra_values

def MPL_intrusion_roughness(volume, csv_file, mpl, voxel_size=5, region_size=10, from_top=True):
    """
    Calculates the roughness of the MPL surface facing the GDL and quantifies the intrusion.

    Parameters:
    - volume: 3D numpy array of the segmented volume (with classes for pore, MPL, and GDL).
    - mpl: The class value for the MPL layer.
    - voxel_size: The physical size of each voxel (default is 5 microns).
    - region_size: The size of the region to calculate local roughness (default is 10).
    - from_top: Boolean indicating whether the MPL is at the top or bottom of the volume.

    Returns:
    - Ra: Arithmetic mean roughness of the MPL surface facing the GDL.
    - Rq: Root mean square roughness.
    - Ra_std_dev: Standard deviation of local roughness values.
    - Ra_coefficient_of_variation: Coefficient of variation for roughness.
    - avg_thickness: Average thickness of the MPL layer.
    """
    # Isolate MPL layer
    mpl_volume = np.where(volume == mpl, 1, 0)

    # Calculate the average thickness of the MPL along the Z-axis
    mpl_thickness = np.sum(mpl_volume, axis=0)  # Sum along the Z-axis
    avg_thickness = int(np.mean(mpl_thickness))
    print('mpl Thickness',avg_thickness)
    # Focus on the surface facing the GDL
    if from_top:
        surface_slice = np.max(np.where(mpl_volume == 1)[0])  # Bottom surface if MPL is on top
    else:
        surface_slice = np.min(np.where(mpl_volume == 1)[0])  # Top surface if MPL is on bottom

    # Fill cracks below the average thickness with 1 (solid)
    if from_top:
        filled_mpl_volume = np.copy(mpl_volume)
        filled_mpl_volume[:avg_thickness, :, :] = 1  # Fill the MPL above the surface

    else:
        filled_mpl_volume = np.copy(mpl_volume)
        filled_mpl_volume[avg_thickness:, :, :] = 1  # Fill the MPL below the surface

    case = 'test'
    visualize_volume(filled_mpl_volume, case, False)
    # Extract the surface using marching cubes
    verts, faces, _, _ = marching_cubes(filled_mpl_volume, level=0)

    # Calculate the global roughness (Ra, Rq) for the relevant surface
    Ra, Rq = calculate_surface_roughness_from_surface(filled_mpl_volume, voxel_size)

    # Calculate local roughness values and standard deviation
    local_Ra_values = calculate_local_roughness(verts, voxel_size, region_size)
    Ra_std_dev = np.std(local_Ra_values)
    Ra_coefficient_of_variation = Ra_std_dev / Ra  # Coefficient of variation
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### MPL/GDL Intrusion Analysis #####'])
        writer.writerow(['Ra', Ra])
        writer.writerow(['Rq', Rq])
        writer.writerow(['Ra_std_dev', Ra_std_dev])
        writer.writerow(['Ra_coefficient_of_variation', Ra_coefficient_of_variation]) 
        writer.writerow(['avg_thickness', avg_thickness])           
    return Ra, Rq, Ra_std_dev, Ra_coefficient_of_variation, avg_thickness

def MPL_count_touching_voxels(volume, csv_file, mpl_class=2, fiber_class=1):
    """
    Counts the number of voxels where the MPL class is adjacent to the fiber class.

    Parameters:
    - volume: 3D numpy array representing the segmented volume.
    - mpl_class: The integer value representing the MPL class (default is 1).
    - fiber_class: The integer value representing the fiber class (default is 2).

    Returns:
    - touching_voxel_count: The number of voxels where the MPL and fiber classes are touching each other.
    """

    # Create binary masks for MPL and fiber classes
    mpl_mask = (volume == mpl_class)
    fiber_mask = (volume == fiber_class)

    # Dilate the MPL mask to find the neighboring voxels
    dilated_mpl_mask = binary_dilation(mpl_mask)

    # Check where dilated MPL mask overlaps with fiber voxels
    touching_voxels = dilated_mpl_mask & fiber_mask

    # Count the number of touching voxels
    touching_voxel_count = np.sum(touching_voxels)

    print('MPL voxels touching GDL: ', touching_voxel_count)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### MPL voxels touching GDL #####'])
        writer.writerow(['MPL_GDL_contact_voxels', touching_voxel_count])

    return touching_voxel_count

def create_network_from_image(binary_image):
    """
    Create a network from the binary image using the SNOW algorithm in PoreSpy
    and OpenPNM's network creation method for PoreSpy.
    """
    # Use PoreSpy's SNOW algorithm to analyze the image
    snow_output = ps.networks.snow2(binary_image)

    # Create an OpenPNM project
    proj = op.Project()

    # Create a network from the PoreSpy output
    network = op.io.network_from_porespy(snow_output)

    return network

def setup_permeability_simulation(network):
    """
    Set up a permeability simulation on the network.
    """
    proj = op.Project()
    geom = op.geometry.GenericGeometry(network=network, pores=network.Ps, throats=network.Ts)
    air = op.phases.Air(network=network)
    
    phys = op.physics.Standard(network=network, phase=air, geometry=geom)

    # Defining StokesFlow algorithm
    flow = op.algorithms.StokesFlow(network=network, phase=air)
    flow.set_value_BC(pores=network.pores('left'), values=1)
    flow.set_value_BC(pores=network.pores('right'), values=0)
    
    # Run the permeability calculation
    flow.run()

    permeability = flow.calc_effective_permeability(domain_area=1.0, domain_length=1.0)
    return permeability

def calculate_permeability(volume):
    """
    Calculate the permeability of a segmented GDL using openPNM.
    """
    # Load the segmented volume
    segmented_volume = volume
    
    # Ensure the GDL is segmented (1 = GDL, 0 = pores)
    binary_image = np.where(segmented_volume == 2, 1, 0)
    
    # Create the pore network
    network = create_network_from_image(binary_image)
    
    # Run permeability simulation
    permeability = setup_permeability_simulation(network)
    
    return permeability

def tortuosity_simulation(binary_volume, csv_file):
    tortuosity = ps.simulations.tortuosity_fd(binary_volume, axis=0) #For toray120 1.85 took 6 h
    print("Tortuosity:", tortuosity)

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write porosity
        writer.writerow(['##### Simulated Tortuosity #####'])
        writer.writerow(['tortuosity', tortuosity])
    return tortuosity

# ######## TEST FUNCTIONS  ########
# filepath = 'C:/Users/a.colliard/Downloads/39bb2_HRNET_HRNET_multi_fusev2_dataset_noaug_-0.6808-23.keras.tif'
# #'C:/Users/a.colliard/Desktop/aimys_project/CT_crops/new/totake/Toray1202.tif'
# #'C:/Users/a.colliard/Downloads/toray1202_fusev2_HRNET_HRNET_fusev2_dataset_noaug_-0.6439-55.keras.tif'

# csv_file = './functions/csv_test.csv'
# binary_image_3d = open_tiff_stack(filepath)
# output_csv_path = './functions/test_csv.csv'
# # Calculate porosity, psd and binary img
# porosity,results, avg_pore, sd = calculate_psd(binary_image_3d, csv_file, voxel_size=5)

# # Calculate surface roughness
# Ra, Rq = calculate_surface_roughness_from_surface(binary_image_3d)
# #MPL_roughness_calculation(binary_image_3d, mpl=2)
# print(f"Arithmetic Mean Roughness (Ra): {Ra}")
# print(f"Root Mean Square Roughness (Rq): {Rq}")
# # Example usage:
# Ra, Rq, Ra_std_dev, Ra_CoV, avg_thickness = MPL_intrusion_roughness(binary_image_3d, csv_file, mpl=2, voxel_size=5)
# print(f"Global Roughness Ra: {Ra}, Rq: {Rq}")
# print(f"Standard Deviation of Local Ra: {Ra_std_dev}")
# print(f"Coefficient of Variation of Local Ra: {Ra_CoV}")
# print(f"Average Thickness: {avg_thickness}")

# # Calculate tortuosity
# tortuosity = 1.85
# #tortuosity_simulation(binary_image_3d, csv_file)

# # Estimate permeability
# ssa = calculate_ssa(binary_image_3d)
# # print(ssa)
# porosity = 0.73
# # #results = {'bin_centers':1, 'pdf':1}
# permeability = estimate_permeability(porosity, csv_file, ssa, tortuosity)
# print(f'Permeability: {permeability}')

# #Calculacte solid surface ratio
# calculate_solid_surface_ratio(binary_image_3d, csv_file, gdl=1, side='bottom')

# #Calculate MPL and GDL thickness
# MPL_GDL_thickness(binary_image_3d, csv_file, axis=0, mpl=2, gdl=1)

# #Calculate MPL crack analysis
# crack_ratio, crack_count, crack_labels, crack_sizes, slice_image = MPL_crack_analysis(binary_image_3d, csv_file)
# print(f"Crack Ratio: {crack_ratio}")
# print(f"Crack Count: {crack_count}")
# print(f"Crack Sizes: {crack_sizes}")
# plot_crack_labels(slice_image, crack_labels)

# # GDL layer PSD calculation
# #layer_pore_size_distribution(binary_image_3d, 1) #Needs furhter logic enhancement

# # MPL heatmap creation
# MPL_heatmap(binary_image_3d,2)

# # Calculate fiber touching MPL voxels
# MPL_count_touching_voxels(binary_image_3d, csv_file, mpl_class=1, fiber_class=2)

# # Calculate permeability via simulation
# #calculate_permeability(binary_image_3d) # Still need to be repaired (Not sure which parameters are relevant)

# # save_results_to_csv(results, porosity, avg_pore, sd, Ra, Rq, permeability, tortuosity, output_csv_path)




