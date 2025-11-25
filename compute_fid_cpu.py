"""
FID/KID computation using CPU-compatible approach for macOS
"""
import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from scipy import linalg
from tqdm import tqdm

# Force CPU for FID computation
device = torch.device('cpu')
print(f'Using device: {device}')

def load_images_from_dir(directory, transform):
    """Load images from directory and apply transform."""
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
    return torch.stack(images)

def get_inception_features(images, model, batch_size=50):
    """Extract features from Inception v3 model."""
    model.eval()
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc='Extracting features'):
            batch = images[i:i+batch_size].to(device)
            feat = model(batch)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    """Calculate FID score between real and fake features."""
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2

    # Compute sqrt of product
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    return float(fid)

def calculate_kid(real_features, fake_features, subset_size=100, num_subsets=10):
    """Calculate KID (Kernel Inception Distance)."""
    n = min(len(real_features), len(fake_features))

    kid_values = []
    for _ in range(num_subsets):
        idx_real = np.random.choice(len(real_features), min(subset_size, n), replace=False)
        idx_fake = np.random.choice(len(fake_features), min(subset_size, n), replace=False)

        real_subset = real_features[idx_real]
        fake_subset = fake_features[idx_fake]

        # Polynomial kernel
        def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
            if gamma is None:
                gamma = 1.0 / X.shape[1]
            return (gamma * np.dot(X, Y.T) + coef0) ** degree

        K_xx = polynomial_kernel(real_subset, real_subset)
        K_yy = polynomial_kernel(fake_subset, fake_subset)
        K_xy = polynomial_kernel(real_subset, fake_subset)

        m = len(real_subset)

        # Compute MMD
        mmd = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1)) + \
              (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) - \
              2 * np.mean(K_xy)

        kid_values.append(mmd)

    return float(np.mean(kid_values) * 100)  # Scale by 100 for readability

def main():
    # Transform for Inception v3 (299x299)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Inception v3 model (pretrained, remove final classification layer)
    print('Loading Inception v3 model...')
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove classification layer
    inception = inception.to(device)
    inception.eval()

    categories = ['cat', 'bus', 'rabbit']
    results = {}

    for category in categories:
        print(f'\n{"="*60}')
        print(f'Computing metrics for {category.upper()}')
        print(f'{"="*60}')

        real_dir = f'./fid_eval/{category}/real'
        fake_dir = f'./fid_eval/{category}/fake'

        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print(f'Directories not found for {category}')
            continue

        real_files = [f for f in os.listdir(real_dir) if f.endswith('.png')]
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.png')]

        print(f'Real images: {len(real_files)}')
        print(f'Fake images: {len(fake_files)}')

        # Load images
        print('Loading real images...')
        real_images = load_images_from_dir(real_dir, transform)

        print('Loading fake images...')
        fake_images = load_images_from_dir(fake_dir, transform)

        # Extract features
        print('Extracting features from real images...')
        real_features = get_inception_features(real_images, inception)

        print('Extracting features from fake images...')
        fake_features = get_inception_features(fake_images, inception)

        print(f'Real features shape: {real_features.shape}')
        print(f'Fake features shape: {fake_features.shape}')

        # Calculate metrics
        print('Calculating FID...')
        fid_score = calculate_fid(real_features, fake_features)

        print('Calculating KID...')
        kid_score = calculate_kid(real_features, fake_features)

        results[category] = {
            'FID': fid_score,
            'KID': kid_score
        }

        print(f'\n{category.upper()} Results:')
        print(f'  FID: {fid_score:.4f}')
        print(f'  KID: {kid_score:.4f}')

    # Print final results
    print('\n' + '='*60)
    print('FINAL EVALUATION RESULTS')
    print('='*60)
    print(f'{"Category":<15} {"FID":<15} {"KID":<15}')
    print('-'*60)

    for category in ['cat', 'bus', 'rabbit']:
        if category in results:
            fid_val = results[category]['FID']
            kid_val = results[category]['KID']
            print(f'{category.capitalize():<15} {fid_val:<15.2f} {kid_val:<15.4f}')

    print('='*60)

    # Save results
    with open('./results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nResults saved to ./results/evaluation_results.json')

if __name__ == '__main__':
    main()
