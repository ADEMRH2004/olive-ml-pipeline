#!/bin/bash

# Script to create GitHub repository and push olive ML pipeline
# Run this after authenticating with: gh auth login

echo "🚀 Creating GitHub repository for Olive ML Pipeline..."

# Create GitHub repository
gh repo create olive-ml-pipeline \
    --description "Machine Learning Pipeline for Olive Grove Classification using Google Earth Engine Data" \
    --public \
    --source=. \
    --remote=origin \
    --push

echo "✅ Repository created and code pushed!"
echo ""
echo "📊 Repository Details:"
echo "   Name: olive-ml-pipeline"
echo "   Description: Machine Learning Pipeline for Olive Grove Classification using Google Earth Engine Data"
echo "   Visibility: Public"
echo ""
echo "🔗 Your repository URL will be displayed above"
echo ""
echo "📁 What's included:"
echo "   • Two-stage LightGBM classification pipeline"
echo "   • IoU metrics and performance visualizations"
echo "   • ResNet and U-Net deep learning templates"
echo "   • PyTorch dataset utilities"
echo "   • Zone-wise analysis for Tunisian olive groves"
echo "   • Training data from Google Earth Engine"
echo ""
echo "🎯 Ready for collaboration and deployment!"