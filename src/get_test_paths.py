import os
import pandas as pd

# 1. Dynamically locate the 'src' folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Step up to the root folder of your project
root_dir = os.path.abspath(os.path.join(script_dir, '..'))

# 3. Path to the validation CSV in the root folder
csv_path = os.path.join(root_dir, 'MURA-v1.1', 'valid_image_paths.csv')

print(f"[SYSTEM] Target Path: {csv_path}")

if not os.path.exists(csv_path):
    print(f"[ERROR] Could not find the MURA folder at the root path: {root_dir}")
    print("Please verify the 'MURA-v1.1' directory is spelled exactly like that in your root folder.")
else:
    try:
        # Load and filter strictly for Hand and Wrist anatomies
        df = pd.read_csv(csv_path, header=None, names=['image_path'])
        test_df = df[df['image_path'].str.contains('XR_HAND|XR_WRIST')].copy()

        # --- THE FIX: PREPEND ROOT DIRECTORY & CONVERT TO WINDOWS PATHS ---
        # This takes 'MURA-v1.1/valid/...' and turns it into 'C:\Users\...\MURA-v1.1\valid\...'
        test_df['absolute_image_path'] = test_df['image_path'].apply(
            lambda x: os.path.normpath(os.path.join(root_dir, x))
        )
        # -----------------------------------------------------------------

        # Save the full absolute paths to a text file inside your 'src' folder
        output_file = os.path.join(script_dir, 'test_images_list.txt')
        test_df['absolute_image_path'].to_csv(output_file, index=False, header=False)

        print(f"\n==================================================")
        print(f"[SUCCESS] Root MURA dataset accessed successfully!")
        print(f"[SUCCESS] Extracted {len(test_df)} testing images.")
        print(f"[SUCCESS] Clean list with full absolute paths saved to:\n -> {output_file}")
        print(f"==================================================")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process dataset CSV. Details: {e}")