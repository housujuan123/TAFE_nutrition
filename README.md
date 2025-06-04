# Ingredient-Guided RGB-D Fusion Network for Nutritional Assessment
This is the official implementation of the paper "[Ingredient-Guided RGB-D Fusion Network for Nutritional Assessment]". 

We propose a novel ingredient-guided RGB-D fusion network that integrates RGB images with depth maps and enables more reliable nutritional assessment guided by ingredient information. Specifically, the multifrequency bimodality fusion module is designed to leverage the correlation between the RGB image and the depth map within the frequency domain. Furthermore, the progressive-fusion module and ingredient-guided module leverage ingredient information to explore the potential correlation between ingredients and nutrients, thereby enhancing the guidance for nutritional assessment learning. We evaluate our approach on a variety of ablation settings on Nutrition5k, where it consistently outperforms state-of-the-art methods.

![image](https://github.com/user-attachments/assets/4dcbc904-f124-4144-bbec-c41c8b899a98)

# Config
Dataset metadata is stored in  nutrition5k_dataset . The organizational form of the dataset is as follows:

|-nutrition5k_dataset
|---imagery
    |---realsense_overhead
            |---Dish1
		|---depth_color.png
		|---rgb.png
            |---Dish2
                |---depth_color.png
		|---rgb.png
            ......
            |---DishM
                |---
Also,The labels for the training and testing sets are as follows:
Training set tags:
rgbd_train_processed.txt  
rgb_in_overhead_train_processed.txt
train_rgb_dish_ingredient1.txt

Testing set tags:
rgbd_test_processed.txt
rgb_in_overhead_test_processed.txt
test_rgb_dish_ingredient_new_1.txt

# Usage
To train the model, use
python train_pretain_swin_convnext_wave_low_depthcat_clip_two_ingr_atten2cat.py

# Citation
@article{feng2024ingredient,
  title={Ingredient-Guided RGB-D Fusion Network for Nutritional Assessment},
  author={Feng, Zhihui and Xiong, Hao and Min, Weiqing and Hou, Sujuan and Duan, Huichuan and Liu, Zhonghua and Jiang, Shuqiang},
  journal={IEEE Transactions on AgriFood Electronics},
  year={2024},
  publisher={IEEE}
}
