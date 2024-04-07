<h2>Tensorflow-Image-Segmentation-Alzheimer-s-Disease (2024/04/08)</h2>

This is an experimental Image Segmentation project for Alzheimer's-Disease based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1yOgBhScahk4yb-xCleNFUfEG3JkXSgwi/view?usp=sharing">
FAZ_Alzheimer-s-Disease-ImageMask-Dataset-V1.zip
</a>
<br>

<br>
Segmentation for test images of 512x512 size by <a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> Model<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/segmentation_samples.png" width="720" height="auto">
<br>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Alzheimer's-Disease Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
<b>OCTA image dataset with pixel-level mask annotation for FAZ segmentation</b><br>

Yufei Wang, Yiqing Shen, Meng Yuan, Jing Xu, Wei Wang and Weijing Cheng<br>
<pre>
https://zenodo.org/records/5075563
</pre>
<pre>
This dataset is publish by the research "A Deep Learning-based Quality Assessment and Segmentation System 
with a Large-scale Benchmark Dataset for Optical Coherence Tomographic Angiography Image"

Detail:
This dataset is the pixel-level mask annotation for FAZ segmentation. 1,101 3 × 3 mm2 sOCTA images chosen 
from gradable and best OCTA images randomly in subset sOCTA-3x3-10k, and 1,143 6 × 6 mm2dOCTA images were 
an notated by an experienced ophthalmologist.
GitHub: https://github.com/shanzha09/COIPS

These datasets are public available, if you use the dataset or our system in your research, 
please cite our paper: 
A Deep Learning-based Quality Assessment and Segmentation System with a Large-scale Benchmark 
Dataset for Optical Coherence Tomographic Angiography Image.
</pre>

Please see also:<br>
<b>Hybrid-FAZ</b>
<pre>
https://github.com/kskim-phd/Hybrid-FAZ
</pre>
<br>


<h3>
<a id="2">
2. Alzheimer's-Disease ImageMask Dataset
</a>
</h3>
 If you would like to train this Alzheimer's-Disease Segmentation model by yourself,
 please download the latest normalized dataset from the google drive 
<a href="https://drive.google.com/file/d/1yOgBhScahk4yb-xCleNFUfEG3JkXSgwi/view?usp=sharing">
FAZ_Alzheimer-s-Disease-ImageMask-Dataset-V1.zip.
</a>
<br>
Please see also:<a href="https://github.com/sarah-antillia/ImageMask-Dataset-FAZ_Alzheimer-s-Disease">ImageMask-Dataset-FAZ_Alzheimer-s-Disease</a>
<br>

<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be
<pre>
./dataset
└─Alzheimer's-Disease
   ├─test
   │  ├─images
   │  └─masks
   ├─train
   │  ├─images
   │  └─masks
   └─valid
       ├─images
       └─masks
</pre>
<b>Alzheimer's-Disease Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/Alzheimer's-Disease_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is not necessarily large.<br>

<br>
<b>train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h2>
3. Train TensorflowUNet Model
</h2>
 We have trained Alzheimer's-Disease TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
, where train_eval_infer.config is the following.<br>
<pre>
; train_eval_infer.config
; 2024/04/08 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = False
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
num_classes    = 1
base_filters   = 16
base_kernels   = (5,5)
num_layers     = 7
dropout_rate   = 0.08
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
;loss          = "bce_iou_loss"
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
;metrics        = ["iou_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
patience      = 10
;metrics       = ["iou_coef", "val_iou_coef"]
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Alzheimer's-Disease/train/images/"
mask_datapath  = "../../../dataset/Alzheimer's-Disease/train/masks/"
create_backup  = False
learning_rate_reducer = True
reducer_patience      = 5
save_weights_only     = True

[eval]
image_datapath = "../../../dataset/Alzheimer's-Disease/valid/images/"
mask_datapath  = "../../../dataset/Alzheimer's-Disease/valid/masks/"

[test] 
image_datapath = "../../../dataset/Alzheimer's-Disease/test/images/"
mask_datapath  = "../../../dataset/Alzheimer's-Disease/test/masks/"

[infer] 
images_dir    = "../../../dataset/Alzheimer's-Disease/test/images/"
output_dir    = "./test_output"
merged_dir    = "./test_output_merged"
binarize      = True

[segmentation]
colorize      = False
black         = "black"
white         = "green"
blursize      = None

[mask]
blur      = True
blur_size = (3,3)
binarize  = False
threshold = 128
</pre>

The training process has just been stopped at epoch 42 by an early-stopping callback as shown below.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_console_output_at_epoch_42.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Alzheimer's-Disease.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/evaluate_console_output_at_epoch_42.png" width="720" height="auto">
<br><br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/evaluation.csv">evaluation.csv</a><br>
The loss score (bce_dice_loss) for this test dataset is very low, and binary_accuracy is also very high.
as shown below.
<pre>
loss,0.0584
binary_accuracy,0.9936
</pre>

<h2>
5. Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Alzheimer's-Disease.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
test_images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/test_images.png" width="1024" height="auto"><br>
test_mask(ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/test_masks.png" width="1024" height="auto"><br>
<hr>

Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/test_output.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/test_output_merged.png" width="1024" height="auto"><br> 
<hr>

Enlarged samples<br>
<table>
<tr>
<td>
test/image deep_21.jpg<br>
<img src="./dataset/Alzheimer's-Disease/test/images/deep_21.jpg" width="512" height="auto">

</td>
<td>
Inferred merged deep_21.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/test_output_merged/deep_21.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/image deep_820.jpg<br>
<img src="./dataset/Alzheimer's-Disease/test/images/deep_820.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Gdeep_820.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/test_output_merged/deep_820.jpg" width="512" height="auto">
</td> 
</tr>



<tr>
<td>
test/image deep_1038.jpg<br>
<img src="./dataset/Alzheimer's-Disease/test/images/deep_1038.jpg" width="512" height="auto">

</td>
<td>
Inferred merged deep_1038.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/test_output_merged/deep_1038.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
test/image superficial_233.jpg<br>
<img src="./dataset/Alzheimer's-Disease/test/images/superficial_233.jpg" width="512" height="auto">

</td>
<td>
Inferred merged superficial_233.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/test_output_merged/superficial_233.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/image superficial_866.jpg<br>
<img src="./dataset/Alzheimer's-Disease/test/images/superficial_866.jpg" width="512" height="auto">

</td>
<td>
Inferred merged superficial_866.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/test_output_merged/superficial_866.jpg" width="512" height="auto">
</td> 
</tr>


</table>
<br>

<h3>
References
</h3>

<b>1.OCTA image dataset with pixel-level mask annotation for FAZ segmentation</b><br>
Yufei Wang, Yiqing Shen, Meng Yuan, Jing Xu, Wei Wang and Weijing Cheng<br>
<pre>
https://zenodo.org/records/5075563
</pre>
<b>2. Hybrid-FAZ </b><br>
Kyungsu Kim<br>
<pre>
https://github.com/kskim-phd/Hybrid-FAZ
</pre>
