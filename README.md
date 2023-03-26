# Brain-Tumor-Segmentation-using-modified-Double-Unet
In this work, we present a deep learning approach for segmenting brain tumors. The data is being used for segmentation and is accomplished through the use of a modified Double U-net design.

**Checkout the paper published in Scientific.net 
https://www.scientific.net/AST.124.111**

Due to the general complexity of MRI brain imaging, brain tumor detection is a difficult problem to solve, but it aims to detect tumors by segmenting them utilizing AI-based algorithms. In this paper, we proposed a segmentation model using a Modified Double U-Net which has improved segmentation accuracy. The proposed model was tested using data from the Kaggle repository and was experimented with different variables of which the results are given in the performance analysis.

Initially, the proposed model was tested with various data splits. From which it was evident that the model outperformed for the 80/20 split. 

![Capture](https://user-images.githubusercontent.com/62705784/227796774-04c0a395-777f-4cf8-a6b2-05d4eec52994.PNG)

It was later experimented with different ASPP filters for which the data split was kept as 80/20 and was discovered that the filter size 64 performed better than the other filter sizes.

![Capture](https://user-images.githubusercontent.com/62705784/227796830-c49d186a-a066-4322-9d04-bb4d17c79fda.PNG)

The model was then tested for different optimizers and Adamax outperformed every other optimizer.

![Capture](https://user-images.githubusercontent.com/62705784/227796888-7e4fbd27-4757-42e3-987c-a1906c534db3.PNG)

Additionally, it was evaluated with various convolutional block and filter sizes and it was determined that the optimal configuration is two blocks of conv2D with (3,3) filters.

![Capture](https://user-images.githubusercontent.com/62705784/227796943-26fa9701-8156-4836-80f1-dc8029389183.PNG)

From the performance analysis it is clear that, in contrast to comparable cutting-edge approaches, our architecture generated more efficient, improved, and robust outcomes. This segmentation algorithm has the potential to be used in real-time medical imaging collections in the future.
