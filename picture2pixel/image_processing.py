import imageio.v2 as imageio
from PIL import Image
import numpy as np
import os

def process_image(filename, width, height, r):
    img = imageio.imread(filename)
    img = np.array(Image.fromarray(img).resize((width, height)))

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if r == 0:
        return img

    R_channel = img[:, :, 0]
    G_channel = img[:, :, 1]
    B_channel = img[:, :, 2]

    U_R, S_R, Vt_R = np.linalg.svd(R_channel, full_matrices=False)
    U_G, S_G, Vt_G = np.linalg.svd(G_channel, full_matrices=False)
    U_B, S_B, Vt_B = np.linalg.svd(B_channel, full_matrices=False)

    R_reconstructed = np.dot(U_R[:, :r], np.dot(np.diag(S_R[:r]), Vt_R[:r, :]))
    G_reconstructed = np.dot(U_G[:, :r], np.dot(np.diag(S_G[:r]), Vt_G[:r, :]))
    B_reconstructed = np.dot(U_B[:, :r], np.dot(np.diag(S_B[:r]), Vt_B[:r, :]))

    R_reconstructed = np.clip(R_reconstructed, 0, 255)
    G_reconstructed = np.clip(G_reconstructed, 0, 255)
    B_reconstructed = np.clip(B_reconstructed, 0, 255)

    reconstructed_image = np.zeros(img.shape)
    reconstructed_image[:, :, 0] = R_reconstructed
    reconstructed_image[:, :, 1] = G_reconstructed
    reconstructed_image[:, :, 2] = B_reconstructed

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

def process_folder(input_folder, output_folder, width, height, r):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subdir_path):
            output_subdir_path = os.path.join(output_folder, subdir)
            if not os.path.exists(output_subdir_path):
                os.makedirs(output_subdir_path)
            
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    processed_image = process_image(file_path, width, height, r)
                    output_file_path = os.path.join(output_subdir_path, filename)
                    imageio.imwrite(output_file_path, processed_image)
                    print(f"Processed and saved: {output_file_path}")


input_folder = 'train'  
output_folder = 'train_new1'  
width = 249  
height = 249  
r = 30

process_folder(input_folder, output_folder, width, height, r)