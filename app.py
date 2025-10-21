import os
import base64
import io
import logging
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-gan-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Try to import TensorFlow with better error handling
TF_AVAILABLE = False
TF_VERSION = "Not available"
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
    logger.info(f"✅ TensorFlow {TF_VERSION} successfully imported")
    
    # Test TensorFlow functionality
    hello = tf.constant('Hello, TensorFlow!')
    logger.info(f"✅ TensorFlow test: {hello.numpy().decode('utf-8')}")
    
except ImportError as e:
    logger.warning(f"❌ TensorFlow import failed: {e}")
except Exception as e:
    logger.warning(f"❌ TensorFlow initialization failed: {e}")

class TrainedGANManager:
    """Manager for your trained GAN models"""
    
    def __init__(self):
        self.available_models = self._discover_trained_models()
        self.loaded_models = {}
    
    def _discover_trained_models(self):
        """Discover available trained models in the models directory"""
        models_dir = 'models'
        available_models = {}
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info("📁 Created models directory")
            return available_models
        
        # Look for trained model files
        model_files = [
            'medical_gan_generator.h5',
            'generator.h5', 
            'model.h5',
            'trained_generator.h5'
        ]
        
        for filename in model_files:
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                model_name = filename.replace('.h5', '').replace('_', ' ').title()
                available_models[filename] = {
                    'path': model_path,
                    'description': f'Trained {model_name}',
                    'name': model_name,
                    'file_size': f"{os.path.getsize(model_path) / (1024*1024):.1f} MB"
                }
                logger.info(f"✅ Found trained model: {filename} ({available_models[filename]['file_size']})")
        
        # If no trained models found, provide demo options
        if not available_models:
            available_models = {
                'demo_chest': {
                    'description': 'Chest X-ray GAN (Demo)', 
                    'name': 'Demo Chest',
                    'file_size': 'Demo'
                },
                'demo_brain': {
                    'description': 'Brain MRI GAN (Demo)', 
                    'name': 'Demo Brain',
                    'file_size': 'Demo'
                }
            }
            logger.info("ℹ️ No trained models found, using demo modes")
        
        return available_models
    
    def load_model(self, model_key):
        """Load a specific trained model"""
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not TF_AVAILABLE:
            logger.warning("❌ TensorFlow not available, cannot load model")
            return None
        
        try:
            if model_key in self.available_models and 'path' in self.available_models[model_key]:
                model_path = self.available_models[model_key]['path']
                logger.info(f"🔄 Loading model from: {model_path}")
                
                # Load the model
                generator = tf.keras.models.load_model(model_path)
                self.loaded_models[model_key] = generator
                
                # Log model details
                input_shape = generator.input_shape
                output_shape = generator.output_shape
                logger.info(f"✅ Successfully loaded model: {model_key}")
                logger.info(f"📊 Model input shape: {input_shape}")
                logger.info(f"📊 Model output shape: {output_shape}")
                
                return generator
            else:
                logger.warning(f"⚠️ Model {model_key} not found or is demo")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading model {model_key}: {e}")
            return None
    
    def generate_with_model(self, model_key, num_images=10, seed=None):
        """Generate images using the trained model"""
        if seed:
            np.random.seed(seed)
        
        # Try to use real trained model first
        generator = self.load_model(model_key)
        
        if generator is not None and TF_AVAILABLE:
            try:
                # Get the latent dimension from the model
                latent_dim = generator.input_shape[1]
                logger.info(f"🎨 Generating {num_images} images with latent dim: {latent_dim}")
                
                # Generate noise
                noise = np.random.normal(0, 1, (num_images, latent_dim))
                
                # Generate images
                generated_images = generator.predict(noise, verbose=0)
                
                logger.info(f"✅ Generated {num_images} images using trained model")
                logger.info(f"📊 Output shape: {generated_images.shape}")
                
                return generated_images
            except Exception as e:
                logger.error(f"❌ Error generating with model: {e}")
                # Fall back to demo generation
                pass
        
        # Fallback to demo generation
        logger.info(f"🔄 Using demo generation for {model_key}")
        return self._demo_generation(model_key, num_images)
    
    def _demo_generation(self, model_key, num_images):
        """Demo generation when no trained model is available"""
        images = []
        for i in range(num_images):
            if 'chest' in model_key.lower():
                img = self._generate_demo_chest_xray()
            elif 'brain' in model_key.lower():
                img = self._generate_demo_brain_mri()
            else:
                img = self._generate_demo_medical_image()
            images.append(img)
        
        return np.array(images).reshape(num_images, 28, 28, 1)
    
    def _generate_demo_chest_xray(self):
        """Demo chest X-ray generation"""
        img = np.random.rand(28, 28) * 0.3
        # Add simple lung structures
        y, x = np.ogrid[:28, :28]
        left_lung = ((x - 8)**2 / 25 + (y - 14)**2 / 36) <= 1
        right_lung = ((x - 20)**2 / 25 + (y - 14)**2 / 36) <= 1
        img[left_lung | right_lung] += 0.4
        
        # Add ribs
        for i in range(3):
            y_pos = 10 + i * 4
            rib_mask = (np.abs(np.arange(28)[:, None] - y_pos) < 0.8) & (np.arange(28) > 5) & (np.arange(28) < 23)
            img[rib_mask] += 0.2
            
        return np.clip(img, 0, 1) * 2 - 1
    
    def _generate_demo_brain_mri(self):
        """Demo brain MRI generation"""
        img = np.random.rand(28, 28) * 0.4
        y, x = np.ogrid[:28, :28]
        brain_mask = ((x - 14)**2 / 64 + (y - 14)**2 / 49) <= 1
        img[brain_mask] += 0.5
        
        # Add ventricles
        vent_mask = ((x - 14)**2 / 16 + (y - 14)**2 / 9) <= 1
        img[vent_mask] -= 0.2
        
        return np.clip(img, 0, 1) * 2 - 1
    
    def _generate_demo_medical_image(self):
        """Generic demo medical image"""
        img = np.random.rand(28, 28) * 0.4
        # Add some structure
        y, x = np.ogrid[:28, :28]
        structure = ((x - 14)**2 / 25 + (y - 14)**2 / 25) <= 1
        img[structure] += 0.3
        
        # Add texture
        img += np.random.rand(28, 28) * 0.1
        
        return np.clip(img, 0, 1) * 2 - 1

# Global GAN manager
gan_manager = TrainedGANManager()

# Available datasets (for UI consistency)
DATASETS = {
    'ChestMNIST': 'Chest X-ray Images',
    'PathMNIST': 'Pathology Images', 
    'BrainMNIST': 'Brain MRI Images',
    'CustomDataset': 'Your Trained Data'
}

@app.route('/')
def index():
    """Home page"""
    # Safely get the first model description
    models_dict = gan_manager.available_models
    first_model_desc = "Select a trained model"
    if models_dict:
        first_key = list(models_dict.keys())[0]
        first_model_desc = models_dict[first_key].get('description', 'Trained Model')
    
    return render_template('index.html', 
                         datasets=DATASETS,
                         models=models_dict,
                         tf_available=TF_AVAILABLE,
                         tf_version=TF_VERSION,
                         first_model_desc=first_model_desc)

@app.route('/generate', methods=['POST'])
def generate_images():
    """Generate synthetic medical images using trained model"""
    try:
        dataset = request.form.get('dataset', 'ChestMNIST')
        model_key = request.form.get('model')
        
        # Safely get model key
        if not model_key and gan_manager.available_models:
            model_key = list(gan_manager.available_models.keys())[0]
        elif not model_key:
            model_key = 'demo_chest'
            
        num_images = int(request.form.get('num_images', 10))
        seed = request.form.get('seed')
        
        if seed and seed.strip():
            seed = int(seed)
        else:
            seed = None
        
        logger.info(f"🚀 Starting generation: {num_images} images from {model_key}, seed: {seed}")
        
        # Generate images using the trained model
        generated_images = gan_manager.generate_with_model(
            model_key=model_key,
            num_images=num_images,
            seed=seed
        )
        
        # Convert to base64
        image_data = []
        for i in range(num_images):
            img_base64 = array_to_base64(generated_images[i])
            image_data.append({
                'id': i,
                'data': f"data:image/png;base64,{img_base64}"
            })
        
        # Create grid
        if num_images > 1:
            cols = min(num_images, 5)
            rows = (num_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            if rows > 1 or cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            for i in range(num_images):
                axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
                axes[i].set_title(f'{i+1}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_images, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            grid_base64 = plot_to_base64(fig)
            grid_data = f"data:image/png;base64,{grid_base64}"
        else:
            grid_data = image_data[0]['data']
        
        # Get model info
        model_info = gan_manager.available_models.get(model_key, {})
        model_name = model_info.get('name', model_key)
        model_description = model_info.get('description', 'Trained Model')
        
        # Check if using trained model
        using_trained = gan_manager.load_model(model_key) is not None
        
        logger.info(f"✅ Generation completed: {num_images} images, using trained model: {using_trained}")
        
        return jsonify({
            'success': True,
            'images': image_data,
            'grid': grid_data,
            'dataset': dataset,
            'model': model_name,
            'model_description': model_description,
            'num_generated': num_images,
            'seed_used': seed,
            'using_trained_model': using_trained
        })
        
    except Exception as e:
        logger.error(f"❌ Error generating images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_image/<int:image_id>', methods=['POST'])
def download_image(image_id):
    """Download a specific generated image"""
    try:
        dataset = request.form.get('dataset', 'ChestMNIST')
        model_key = request.form.get('model')
        
        if not model_key and gan_manager.available_models:
            model_key = list(gan_manager.available_models.keys())[0]
        elif not model_key:
            model_key = 'demo_chest'
            
        seed = request.form.get('seed')
        
        if seed and seed.strip():
            seed = int(seed)
        else:
            seed = None
        
        # Regenerate the specific image
        generated_images = gan_manager.generate_with_model(
            model_key=model_key,
            num_images=image_id + 1,
            seed=seed
        )
        
        if len(generated_images) <= image_id:
            return jsonify({'error': 'Image not found'}), 404
        
        # Convert to downloadable format
        image_array = generated_images[image_id]
        image = ((image_array.squeeze() + 1) * 127.5).astype(np.uint8)
        pil_img = Image.fromarray(image, mode='L')
        
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        
        model_info = gan_manager.available_models.get(model_key, {})
        model_name = model_info.get('name', model_key)
        filename = f"{dataset}_{model_name}_{image_id + 1}.png"
        
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"❌ Error downloading image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Check status of trained models"""
    status = {
        'tensorflow_available': TF_AVAILABLE,
        'tensorflow_version': TF_VERSION,
        'available_models': gan_manager.available_models,
        'loaded_models': list(gan_manager.loaded_models.keys())
    }
    return jsonify(status)

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

def array_to_base64(image_array):
    """Convert numpy array to base64 image"""
    # Handle both [0,1] and [-1,1] ranges
    if image_array.min() >= -1 and image_array.max() <= 1:
        image = ((image_array.squeeze() + 1) * 127.5).astype(np.uint8)
    else:
        image = (image_array.squeeze() * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(image, mode='L')
    
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("Synthetic Medical Image Generator (MedMNIST)")
    print("=" * 60)
    print(f"TensorFlow Available: {TF_AVAILABLE}")
    print(f"TensorFlow Version: {TF_VERSION}")
    print("Available Models:")
    for model_key, model_info in gan_manager.available_models.items():
        size_info = model_info.get('file_size', 'Unknown')
        print(f"  - {model_key}: {model_info.get('description', 'No description')} ({size_info})")
    print("=" * 60)
    print("Access: http://localhost:5000")
    print("Model Status: http://localhost:5000/model_status")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)