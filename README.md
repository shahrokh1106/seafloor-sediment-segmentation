# seafloor-sediment-segmentation
Tool-Assisted Annotation of Seafloor Sediment-linked Features Using Weakly Supervised Semantic Segmentation

Pixel-wise labeling of seafloor imagery is highly time-consuming, limiting the scalability of benthic habitat monitoring. While existing underwater computer vision research has largely focused on visually prominent habitats, sediment-linked benthic features remain underexplored and lack annotated datasets. To address this gap, we propose a tool-assisted annotation framework based on weakly supervised semantic segmentation. The framework follows a three-phase pipeline: feature-based pseudo-mask generation, binary class-specific segmentation, and iterative multiclass segmentation with pseudo-mask expansion and affinity-field regularization. Applied to six ecologically important sediment features, the approach progressively improves pseudo-label quality while reducing reliance on dense expert annotations, providing a scalable solution that can accelerate the annotation of seafloor sediment features.


# Installation
First install Pytorch. Depending on the GPU capability of your system, you can install it from https://pytorch.org/get-started/locally/
## Install FeatUp from Source
To install FeatUp in editable mode (for development or customization), run:

```bash
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .
```

# TO BE UPDATED ...