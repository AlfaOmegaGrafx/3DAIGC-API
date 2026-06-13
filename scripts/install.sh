#!/bin/bash
#
# ------------------------------------------------------------------------------
# Realistic full stack (TRELLIS / spconv / kaolin / all thirdparty builds)
#
# Prefer the container path from the repo root (NVIDIA Container Toolkit required):
#   docker compose up -d --build
#
# The Dockerfile uses CUDA 12.4 devel + Miniconda on Ubuntu 20.04 and matches the
# wheels and compile steps this script expects (x86_64). See README "Docker".
#
# This bare-metal script targets an already-activated Conda env on Linux x86_64
# with CUDA compatible with PyTorch cu124. Uncomment the "conda create" line
# below if you want this script to create `3daigc-api` for you.
#
# aarch64 (e.g. DGX Spark): many pip wheels here are x86_64-only. Use an ARM-capable
# image/pipeline, or a cloud GPU template (e.g. RunPod in README), unless you
# maintain your own aarch64 builds. Local venv: ./scripts/install_local_venv_extras.sh
# (or ./scripts/bootstrap_local_models.sh then ./scripts/install_next_steps.sh); for
# each shell: source ./scripts/env_local_gpu.sh then ./scripts/run_local_venv.sh
#
# Before either path:  git submodule update --init --recursive
# (bootstrap script runs this for you.)
# ------------------------------------------------------------------------------

echo "========================================"
echo "Starting 3DAIGC-API Installation"
echo "========================================"
echo "The installation may take a while, please wait..."
echo ""

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH."
    echo "        Use Docker (recommended):  docker compose up -d --build"
    echo "        Or install Miniconda and re-run from an activated env."
    exit 1
fi

echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-"(none — activate 3daigc-api first)"}"
# To create the env automatically, uncomment:
# conda create -n 3daigc-api python=3.10 -y
# conda activate 3daigc-api

echo "[INFO] Using python: $(command -v python || true) ($(python -V 2>/dev/null || true))"

echo "[INFO] Installing PyTorch with CUDA 12.4 support..."
## install pytorch for specific cuda versions
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PyTorch installation completed"
else
    echo "[ERROR] Failed to install PyTorch"
    exit 1
fi

echo ""
echo "========================================"
echo "Installing TRELLIS Dependencies"
echo "========================================"
### we startup with the environment of trellis ###
echo "[INFO] Changing directory to thirdparty/TRELLIS..."
cd thirdparty/TRELLIS.2
echo "[INFO] Running TRELLIS.2 setup script..."
. ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu124.html
if [ $? -eq 0 ]; then
    echo "[SUCCESS] TRELLIS setup completed"
else
    echo "[ERROR] TRELLIS setup failed"
    exit 1
fi

echo "[INFO] Installing TRELLIS(v1) requirements on top of TRELLIS.2..."
pip install pymeshfix igraph 
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# for systems with glibc < 2.29 , you may need to build kaolin from source manually
echo "[NOTE] For systems with glibc < 2.29, you may need to build kaolin from source manually"

echo ""
echo "========================================"
echo "Installing PartField Dependencies"
echo "========================================"
# install PartField for mesh segmentation 
echo "[INFO] Changing directory to thirdparty/PartField..."
cd ../../thirdparty/PartField 
echo "[INFO] Installing PartField core dependencies..."
pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PartField core dependencies installed"
else
    echo "[ERROR] Failed to install PartField core dependencies"
    exit 1
fi

echo "[INFO] Installing additional PartField dependencies..."
pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Additional PartField dependencies installed"
else
    echo "[ERROR] Failed to install additional PartField dependencies"
    exit 1
fi

echo "[INFO] Installing PyTorch Geometric extensions..."
pip install torch-scatter torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PyTorch Geometric extensions installed"
else
    echo "[ERROR] Failed to install PyTorch Geometric extensions"
    exit 1
fi
# installation for PartField end 
echo "[SUCCESS] PartField installation completed"


echo ""
echo "========================================"
echo "Installing Hunyuan3D 2.1 Dependencies"
echo "========================================"
### installation for hunyuan3d 2.1  ###
echo "[INFO] Changing directory to thirdparty/Hunyuan3D-2.1..."
cd ../../thirdparty/Hunyuan3D-2.1
echo "[INFO] Installing custom rasterizer for Hunyuan3D 2.1..."
cd hy3dpaint/custom_rasterizer
pip install -e . --no-build-isolation
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.1 custom rasterizer installed"
else
    echo "[ERROR] Failed to install Hunyuan3D 2.1 custom rasterizer"
    exit 1
fi

echo "[INFO] Building differentiable renderer for Hunyuan3D 2.1..."
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Hunyuan3D 2.1 differentiable renderer built successfully"
else
    echo "[ERROR] Failed to build Hunyuan3D 2.1 differentiable renderer"
    exit 1
fi
cd ../..
echo "[INFO] Installing Hunyuan3D 2.1 requirements..."
pip install -r requirements-inference.txt 
### installation for hunyuan3d 2.1 end ###
echo "[SUCCESS] Hunyuan3D 2.1 installation completed"

echo ""
echo "========================================"
echo "Installing UniRig Dependencies"
echo "========================================"
### unirig for auto-rigging  ###
echo "[INFO] Changing directory to thirdparty/UniRig..."
cd ../../thirdparty/UniRig
echo "[INFO] Installing spconv-cu120 for UniRig..."
pip install spconv-cu120
pip install pyrender fast-simplification python-box timm
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UniRig dependencies installed"
else
    echo "[ERROR] Failed to install UniRig dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "Installing PartPacker Dependencies"
echo "========================================"
### part packer  ###
echo "[INFO] Changing directory to thirdparty/PartPacker..."
cd ../../thirdparty/PartPacker
echo "[INFO] Installing PartPacker requirements..."
pip install pybind11==3.0.1
pip install meshiki kiui fpsample pymcubes einops
if [ $? -eq 0 ]; then
    echo "[SUCCESS] PartPacker requirements installed"
else
    echo "[ERROR] Failed to install PartPacker requirements"
    exit 1
fi
### part packer end ###
echo "[SUCCESS] PartPacker installation completed"

### partuv(requires only bpy, partuv) ###
echo "[INFO] Installing partuv requirements..."
pip install seaborn partuv 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] partuv requirements installed"
else
    echo "[ERROR] Failed to install partuv requirements"
    exit 1
fi
pip install blenderproc 
### partuv end ###

### P3-SAM (Hunyuan3D-Part) ###
echo ""
echo "========================================"
echo "Installing P3-SAM Dependencies"
echo "========================================"
cd ../../thirdparty/Hunyuan3DPart/P3SAM
echo "[INFO] Installing P3-SAM requirements..."
# Install numba for acceleration
pip install numba scikit-learn fpsample
if [ $? -eq 0 ]; then
    echo "[SUCCESS] P3-SAM requirements installed"
else
    echo "[ERROR] Failed to install P3-SAM requirements"
    exit 1
fi
### P3-SAM end ###

### FastMesh ###
cd ../../../thirdparty/FastMesh 
echo "[INFO] Installing FastMesh requirements..."
pip install -r requirement_extra.txt
if [ $? -eq 0 ]; then
    echo "[SUCCESS] FastMesh requirements installed"
else
    echo "[ERROR] Failed to install FastMesh requirements"
    exit 1
fi
### FastMesh end ###

### UltraShape ###
echo ""
echo "========================================"
echo "Installing UltraShape Dependencies"
echo "========================================"
cd ../../../thirdparty/UltraShape
echo "[INFO] Installing UltraShape requirements..."
# pip install -r requirements.txt
# actually only cubvh is required based besides trellis.2 env  
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
if [ $? -eq 0 ]; then
    echo "[SUCCESS] UltraShape requirements installed"
else
    echo "[ERROR] Failed to install UltraShape requirements"
    exit 1
fi
### UltraShape end ###

### VoxHammer ###
echo ""
echo "========================================"
echo "Installing VoxHammer Dependencies"
echo "========================================"
cd ../VoxHammer
echo "[INFO] Installing VoxHammer requirements..."
# pip install -r requirements.txt
# only bpy-renderer and pysdf are required besides trellis.2 env  
pip install git+https://github.com/huanngzh/bpy-renderer.git
pip install pysdf setencepiece
if [ $? -eq 0 ]; then
    echo "[SUCCESS] VoxHammer requirements installed"
else
    echo "[ERROR] Failed to install VoxHammer requirements"
    exit 1
fi
echo "[NOTE] VoxHammer uses TRELLIS pipeline which is already installed"
### VoxHammer end ###

cd ../../

echo ""
echo "========================================"
echo "Installing Project Dependencies"
echo "========================================"
### for this project (fastapi / uvicorn relevant etc.)  ###
echo "[INFO] Installing main project requirements..."
pip install -r requirements.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Main project requirements installed"
else
    echo "[ERROR] Failed to install main project requirements"
    exit 1
fi

echo "[INFO] Installing test requirements..."
# testing 
pip install -r requirements-test.txt 
if [ $? -eq 0 ]; then
    echo "[SUCCESS] Test requirements installed"
else
    echo "[ERROR] Failed to install test requirements"
    exit 1
fi

echo "[INFO] Installing huggingface_hub for model downloading..."
# for downloading models 
pip install huggingface_hub
if [ $? -eq 0 ]; then
    echo "[SUCCESS] huggingface_hub installed"
else
    echo "[ERROR] Failed to install huggingface_hub"
    exit 1
fi

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo "All installation done successfully!"

echo "Checking CUDA availability..."
python -c "import torch; print(torch.cuda.is_available())" && echo "CUDA installed successfully" || echo "Failed"

echo "Checking PyTorch version..."
python -c "import torch; print(torch.__version__)" && echo "PyTorch installed successfully" || echo "Failed"

echo "Checking Blender availability..."
python -c "import bpy" && echo "Blender installed successfully" || echo "Failed"

echo "Checking Other Packages..."
python -c "import kaolin; print(kaolin.__version__)" && echo "Kaolin installed successfully" || echo "Failed"
python -c "import open3d; import pymeshlab" && echo "Open3D and pymeshlab installed successfully" || echo "Failed"

# install other runtime dependencies
apt update
apt install libsm6 libegl1 libegl1-mesa libgl1-mesa-dev -y # for rendering 



