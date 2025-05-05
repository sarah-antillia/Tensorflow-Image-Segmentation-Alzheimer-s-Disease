<h2>Tensorflow-Image-Segmentation-Alzheimer-s-Disease (Updated: 2025/05/06)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is an experimental Image Segmentation project for Alzheimer's-Disease based on
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1yOgBhScahk4yb-xCleNFUfEG3JkXSgwi/view?usp=sharing">
FAZ_Alzheimer-s-Disease-ImageMask-Dataset-V1.zip
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
 The inferred colorized masks predicted by our segmentation model trained on the ImageMaskDataset appear 
 similar to the ground truth masks.
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_34.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_34.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_34.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_55.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_55.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_55.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_199.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_199.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_199.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Alzheimer's-Disease Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other more advanced TensorFlow UNet Models to get better segmentation models:<br>
<br>
<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
<b>OCTA image dataset with pixel-level mask annotation for FAZ segmentation</b><br>

Yufei Wang, Yiqing Shen, Meng Yuan, Jing Xu, Wei Wang and Weijing Cheng<br>
<br>
<a href="https://zenodo.org/records/5075563">https://zenodo.org/records/5075563</a>
<br>
<br>
This dataset is publish by the research "A Deep Learning-based Quality Assessment and Segmentation System<br> 
with a Large-scale Benchmark Dataset for Optical Coherence Tomographic Angiography Image"<br>
<br>
<b>Detail:</b><br>
This dataset is the pixel-level mask annotation for FAZ segmentation. 1,101 3 × 3 mm2 sOCTA images chosen <br>
from gradable and best OCTA images randomly in subset sOCTA-3x3-10k, and 1,143 6 × 6 mm2dOCTA images were <br>
an notated by an experienced ophthalmologist.<br>
GitHub: https://github.com/shanzha09/COIPS<br>
<br>
These datasets are public available, if you use the dataset or our system in your research,<br> 
please cite our paper: <br>
A Deep Learning-based Quality Assessment and Segmentation System with a Large-scale Benchmark<br> 
Dataset for Optical Coherence Tomographic Angiography Image.<br>
<br>
Please see also:<br>
<b>Hybrid-FAZ</b><br>
<a href="https://github.com/kskim-phd/Hybrid-FAZ">
https://github.com/kskim-phd/Hybrid-FAZ
</a>
<br>
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
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
4 Train TensorflowUNet Model
</h3>
 We trained Alzheimer's-Disease TensorflowUNet Model by using the 
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Alzheimer's-Disease and run the following bat file for Python script <a href="./src/TensorflowUNetTrainer.py">TensorflowUNetTrainer.py</a>.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<hr>
<b>Model parameters</b><br>
Defined small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and large num_layers (including a bridge).
<pre>
base_filters   = 16 
base_kernels   = (7,7)
num_layers     = 8
dilation       =(2,2)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.
<pre>
[model]
learning_rate  = 0.0001
</pre>
<b>Online augmentation</b><br>
Disabled our online augmentation. To enable the augmentation, set generator parameter to True.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback. 
<pre> 
[train]
learning_rate_reducer = True
reducer_factor        = 0.4
reducer_patience      = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callback</b><br>
Enabled EpochChange infer callback.<br>
<pre>
[train]
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
num_infer_images       = 6
</pre>

By using this EpochChangeInference callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at start (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>

<b>Epoch_change_inference output at start (49,50,51)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>

In this case, the training process stopped at epoch 51 by EarlyStopping Callback as shown below.<br>
<b>Training console output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/train_console_output_at_epoch_51.png" width="720" height="auto"><br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
5 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Alzheimer's-Disease.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/evaluate_console_output_at_epoch_51.png" width="720" height="auto">
<br><br>

The loss (bce_dice_loss) score for this test dataset is low, but dice_coef high as shown below.<br>
<pre>
loss,0.0618
dice_coef,0.9058
</pre>
<br>

<h3>
6 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Alzheimer's-Disease.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>



<b>Enlarged images and masks (512x512 pixels)</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_21.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_21.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_55.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_55.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_55.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_199.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_199.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_199.jpg" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_273.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_273.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_273.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_322.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_322.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_322.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/images/deep_408.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test/masks/deep_408.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Alzheimer's-Disease/mini_test_output/deep_408.jpg" width="320" height="auto"></td>
</tr>

</table>

<br>
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

