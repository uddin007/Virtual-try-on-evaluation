# Advancing Virtual Try-On: A Multi-Step Evaluation Pipeline 
The rapid advancements in virtual try-on technology have opened new frontiers in digital fashion and e-commerce, yet challenges in garment fidelity and visual realism persist. In this study, we present a comprehensive evaluation pipeline to assess the capabilities and limitations of state-of-the-art try-on models like IDM-VTON. The process begins with generating high-quality, digitally created models in diverse outfits and environments using the black-forest-labs/FLUX.1-dev model, enhanced with the Flux-uncensored-v2 weights. Garments sourced from the internet are seamlessly applied to these models using IDM-VTON, leveraging its advanced diffusion-based approach to ensure authenticity and detail preservation. To further refine the visuals, image resolution is enhanced through the jasperai/Flux.1-dev-ControlNet-Upscaler, creating outputs that meet real-world expectations. This article delves into the strengths and gaps identified in try-on performance, offering insights into the future potential of virtual try-on systems.

<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-2.png" alt="Gen Image 1" width="48%">
  <img src="./images/generated_images/uncensored_image_hf-3.png" alt="Gen Image 2" width="48%">
  <img src="./images/generated_images/uncensored_image_hf-4.png" alt="Gen Image 3" width="48%">
  <img src="./images/generated_images/uncensored_image_hf-17.png" alt="Gen Image 4" width="48%">
</div>

I employed the **black-forest-labs/FLUX.1-dev** model in conjunction with the **enhanceaiteam/Flux-uncensored-v2** weights to generate high-quality digital models as a foundation for virtual try-on workflows. The process leverages state-of-the-art **diffusion-based generative modeling** to synthesize realistic and customizable human models tailored for subsequent garment simulations.

The **black-forest-labs/FLUX.1-dev** model operates as a base architecture optimized for generating detailed human models by employing a latent diffusion process. This approach iteratively refines noise-injected latent representations to recover high-resolution image data, ensuring photorealism and anatomical accuracy. Diffusion models excel in generating nuanced details like skin texture, hair features, and pose alignment by learning the distribution of training data and reversing the noise diffusion process.

To enhance the outputs' versatility and resolution, I integrated the **enhanceaiteam/Flux-uncensored-v2** weights into the pipeline. These weights are pre-trained on diverse datasets emphasizing varied human postures, lighting conditions, and ethnic diversity, augmenting the base model's ability to generalize across scenarios. The weights act as fine-tuned parameters guiding the diffusion process towards more consistent outputs, particularly when handling challenging attributes like occlusions or edge details.

The synthesis begins with a Gaussian noise vector, which progressively transitions into a coherent image through a series of denoising steps. Each step is guided by the learned representations encoded in the model and the fine-tuned weights. This process allows for precise control over attributes like pose, body proportions, and even facial expressions, making it an ideal precursor for try-on applications.

In this project we explored the following Vitrual Try-on models and selected IDM-VTON to explore furhter for it's superior performacne.

* sangyun884/HR-VITON
* rlawjdghek/StableVITON
* Zheng-Chong/CatVTON
* shadow2496/VITON-HD
* yisol/IDM-VTON

The original IDM-VTON article is available here https://arxiv.org/abs/2403.05139 and a summary is provided below:

![idm-vton](https://github.com/user-attachments/assets/9b721fc6-344b-4ad5-8882-ac10cd79cf0c)

The IDM-VTON framework addresses image-based virtual try-on by rendering a person wearing a curated garment given images of both the person and the garment. IDM-VTON introduces a cutting-edge approach to image-based virtual try-on, enabling the creation of realistic images of a person wearing curated garments while maintaining garment fidelity and visual naturalness. Unlike previous methods, which often compromise garment details for improved image quality, IDM-VTON combines the strengths of diffusion-based generative modeling with novel semantic encoding techniques to overcome these challenges. 

At the core of the framework is a diffusion model enhanced by a base UNet architecture, augmented with two key modules that encode garment semantics. High-level semantic features are extracted through a visual encoder and integrated into the cross-attention layer, while low-level garment details are processed through a parallel UNet and fused into the self-attention layer. This dual-level encoding ensures both the overall alignment of the garment with the person and the preservation of intricate textures and details. Additionally, IDM-VTON incorporates detailed natural language descriptions for both garments and person images, leveraging the contextual power of text-to-image diffusion models to enhance the authenticity and personalization of the generated visuals.

A significant innovation in IDM-VTON is its customization mechanism, where fine-tuning the decoder layers of the UNet for specific garment-person image pairs enhances fidelity and robustness, especially in complex real-world scenarios. Experimental evaluations demonstrate that IDM-VTON outperforms both GAN-based and earlier diffusion-based models in generating photorealistic virtual try-on images while maintaining garment and body coherence. Its versatility makes it particularly effective for "in-the-wild" applications involving diverse body poses and garment configurations.

Despite its advancements, IDM-VTON has some limitations. It struggles to preserve fine human attributes, such as tattoos or skin details, in masked regions, suggesting the need for future methods to condition generation on these attributes. Furthermore, while the model effectively utilizes captions to align garment and person images, broader applications for textual prompts, such as controlling specific garment designs, remain unexplored.

Overall, IDM-VTON represents a significant leap forward in virtual try-on technology. By combining sophisticated diffusion modeling, semantic encoding, and real-world adaptability, it delivers highly authentic and customizable try-on experiences. However, its success underscores the importance of responsible use to protect intellectual property and prevent misuse, ensuring the technology is a force for innovation in fashion and e-commerce.

We identified, the most efficient way to run this model is by using the Gradio web application available in the repo (https://github.com/yisol/IDM-VTON). A running version with required packages is attached in this repo. The compute used was Google Cloud A100 GPU. This is a NVIDIA Ampere A100, is the cornerstone of the A2 VM family, designed to excel in high-performance computing (HPC) and demanding applications like machine learning. Boasting 40GB of high-performance HBM2 memory and up to 20 times faster compute performance than its predecessor. With 6,912 FP32/INT32 CUDA cores and 3456 FP64 CUDA cores, it offers robust processing power. The A2 VM family includes A2 Ultra, equipped with A100 80GB GPUs and Local SSD disks, and A2 Standard, featuring A100 40GB GPUs, providing users with flexibility to choose the optimal configuration for their specific needs.

<div style="display: flex; justify-content: space-around;">
  <img src="./images/garments/adidas_WOMEN_Tabela_23_Jersey_Team_Royal_BlueWhite_Front.jpg.webp" alt="Gen Image 1" width="48%">
  <img src="./images/try-on-images/upscaled-67.png" alt="Gen Image 2" width="48%">
  <img src="./images/garments/Fruit-of-the-Loom-Womens-Value-Tank-Top_c5104ab6-c639-4c6e-86e2-762f906e794b.96b55a3015a010a1fef0c263d15e1ee3.jpeg.webp" alt="Gen Image 3" width="48%">
  <img src="./images/try-on-images/upscaled-50.png" alt="Gen Image 4" width="48%">
  <img src="./images/garments/1685456923000-jNZ4eT-2.jpg" alt="Gen Image 5" width="48%">
  <img src="./images/try-on-images/upscaled-56.png" alt="Gen Image 6" width="48%">
  <img src="./images/garments/61wsRxYgOIL._AC_SX679_.jpg" alt="Gen Image 7" width="48%">
  <img src="./images/try-on-images/upscaled-53.png" alt="Gen Image 8" width="48%">
</div>

By default, model outputs are in 675x675 image size. The resolution of those images were increased to 1024x1024 using upscaler. We are utilizing **FluxControlNetPipeline**, leveraging the "jasperai/Flux.1-dev-Controlnet-Upscaler" and "black-forest-labs/FLUX.1-dev" models to achieve high-quality, detailed image upscaling. This pipeline integrates **ControlNet** with a diffusion-based generative process, providing both precision and flexibility in enhancing image resolution. 

The `FluxControlNetModel.from_pretrained` method loads the pre-trained ControlNet model, "jasperai/Flux.1-dev-Controlnet-Upscaler," which is specifically designed to guide the diffusion process for upscaling tasks. ControlNet acts as a conditioning network, enabling precise control over the structure and details of the output image by injecting additional guidance during the generative process. The model's weights are loaded in the **bfloat16** precision format, reducing memory usage while maintaining computational efficiency, especially useful for large-scale upscaling tasks.

The `FluxControlNetPipeline` integrates the pre-trained base model ("black-forest-labs/FLUX.1-dev") with the ControlNet. The base model employs a **latent diffusion process**, which iteratively denoises a latent space representation of the image. The diffusion model learns the image data distribution, enabling the reconstruction of fine details during upscaling. The ControlNet provides additional constraints to the diffusion process, ensuring that the structural fidelity of the original image is preserved, such as edges, textures, and other critical features.

The process can be extended to event specific generated images, for example outdoor and indoor sports. Although upscaling can cause some changes to the generated model. In following images, left is generated, middle is the garment and right is the try-on image. 

<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-31.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/B2115-206-F_2f522a3f-9b54-480e-b78e-8d8e8825d1ce_1024x1024@2x.png.webp" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-42.png" alt="Gen Image 3" width="30%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-24.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/41OanteZpkL._AC_SY445_.jpg" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-29.png" alt="Gen Image 3" width="30%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-19.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/images-2.jpeg" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-31.png" alt="Gen Image 3" width="30%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-30.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/juventus-2425-home-jersey-is8002-1.jpg" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-41.png" alt="Gen Image 3" width="30%">
</div>

### Limitations & Challenges

The IDM-VTON (Image-based Virtual Try-On Network) model, while highly effective in generating virtual try-on results, exhibits notable limitations in accurately capturing fine details from garments, such as text, intricate logos, or small decorative elements. These limitations can be attributed to several factors inherent in its architecture and operational framework:

IDM-VTON typically operates on downscaled image resolutions to optimize computational efficiency and model performance. This downscaling leads to a loss of fine-grained details, particularly in regions where garment patterns, logos, or text are densely packed or intricately designed. The model's warping and texture transfer mechanisms prioritize maintaining the overall structural coherence of the garment. However, fine details such as small fonts, embroidery patterns, or logos may be distorted or blurred during the warping process, especially when adapting the garment to the target body pose or shape. IDM-VTON relies heavily on learning global garment and pose features for accurate alignment and integration. This emphasis on global feature extraction often comes at the expense of localized features, leading to reduced fidelity in regions containing high-frequency details. When parts of the garment overlap with complex body poses or occlusions, the model's ability to retain fine details diminishes further. The reconstructed areas may exhibit artifacts or simplified textures that fail to reflect the original garment details. IDM-VTON's architecture lacks specialized components or loss functions designed explicitly for preserving high-frequency details such as logos or text. Current implementations optimize for general texture consistency rather than targeted high-detail fidelity. Finally, The quality and diversity of the training dataset also play a critical role in the model's performance. If the dataset does not adequately represent garments with high-detail features, the model may fail to generalize well to such scenarios during inference.

<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-28.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/women1.jpg" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-37.png" alt="Gen Image 3" width="30%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-14.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/s-l1200.jpg" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-51.png" alt="Gen Image 3" width="30%">
</div>
<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-21.png" alt="Gen Image 1" width="30%">
  <img src="./images/garments/6651_B-Red_800x.jpg.webp" alt="Gen Image 2" width="30%">
  <img src="./images/try-on-images/upscaled-73.png" alt="Gen Image 3" width="30%">
</div>

The first image showed garment properly attached to the person however, the logo is not properly transfered from the garment to the image. The 2nd and 3rd ones showed some mismatch around upper/lower garment or hand/glove areas. 

### Recommendations for Improvement

To address these limitations and enhance the model's ability to capture fine details, the following strategies could be explored:  

Implementing a high-resolution refinement stage, such as super-resolution modules or conditional GANs, can help recover lost details post-inference. Incorporating loss functions specifically designed to prioritize high-frequency detail retention (e.g., perceptual loss or texture loss) can improve the fidelity of small features. Adding attention mechanisms focused on areas with fine details can ensure better feature preservation during garment warping and texture transfer. Expanding the training dataset with a focus on garments containing high-detail elements such as text, logos, or patterns can improve the model's ability to generalize to similar scenarios. Leveraging hybrid approaches that combine traditional image processing techniques (e.g., edge detection) with deep learning models could enhance fine detail preservation.

