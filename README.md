# Semi-Supervised Semantic Segmentation with Cross-Consistency Training (CCT) 
https://github.com/yassouali/CCT

## Requirement
You can create a new conda environment first. Then inside that environment:
First, you need to install the compatible pytorch and torchvision version following the offical instruction at https://pytorch.org/
Then, you need to install modules in the requirements.txt file by running:
`pip install -r requirements.txt`

## Dataset preparing
The PASCAL VOC dataset can be downloaded at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
You create a folder name `VOCtrainval_11-May-2012` in the `CCT` folder, and extract the downloaded .tar file in the folder `VOCtrainval_11-May-2012`

Then, you have to download data label at https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0.
Extract the downloaded zip file, then move them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`

## Reproduce our best improvement method
Make sure you activate the created environment.

In your terminal, navigate to `CCT` folder, then run the following command:
`python3 CCT_GAN_WSSL.py`

There is also a notebook `CCT_GAN_WSSL.ipynb`. You can install jupyter notebook in the conda environment to run this notebook.

You should see the best validation mIoU at 73.5 at epoch(4) or after the program finishes running

You may also try different backbones implemented for our project. By commenting/uncommenting out the lines in encoder.py, try ```poolformer-m36```  and ```convnext-base-ink22``` architectures to produce better results. Please follow the instructions in ```encoder.py``` comments to implement the architecture change correctly. 

