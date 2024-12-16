from skimage.feature import hog

def extract_with_hog(img, orientation=7, pixels_per_cell=(8, 8), cells_per_block=(4,4)):
    """
    return hog features for an image in vector
    """

    hog_features = hog(
        img,
        orientations=orientation,               
        pixels_per_cell=pixels_per_cell,      
        cells_per_block=cells_per_block,
        block_norm = 'L2-Hys',
        visualize=False,               
        channel_axis=None 
    )

    return hog_features
