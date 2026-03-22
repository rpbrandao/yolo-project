#!/usr/bin/env bash
# setup.sh — Bootstrap the project environment
set -e

echo "🔧 Setting up YOLO Project environment..."

# 1. Python check
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.8+"; exit 1; }

# 2. Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# 3. Create dataset directory structure
echo "📁 Creating dataset directories..."
mkdir -p dataset/images/{train,val,test}
mkdir -p dataset/labels/{train,val,test}
mkdir -p results

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Add your images to   dataset/images/train/"
echo "  2. Annotate with LabelMe: labelme dataset/images/train/"
echo "  3. Convert labels:        python scripts/labelme2yolo.py --input dataset/images/train/ --output dataset/labels/train/ --classes your_class1 your_class2"
echo "  4. Update class names in  dataset/data.yaml"
echo "  5. Split dataset:         python scripts/split_dataset.py --images_dir dataset/images/train/"
echo "  6. Train:                 python scripts/train.py --data dataset/data.yaml"
echo ""
echo "🚀 Or use the Colab notebook: notebooks/train_colab.ipynb"
