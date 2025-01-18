# Advancing Virtual Try-On: A Multi-Step Evaluation Pipeline for Image Generation, Garment Testing, and Resolution Enhancement
The rapid advancements in virtual try-on technology have opened new frontiers in digital fashion and e-commerce, yet challenges in garment fidelity and visual realism persist. In this study, we present a comprehensive evaluation pipeline to assess the capabilities and limitations of state-of-the-art try-on models like IDM-VTON. The process begins with generating high-quality, digitally created models in diverse outfits and environments using the black-forest-labs/FLUX.1-dev model, enhanced with the Flux-uncensored-v2 weights. Garments sourced from the internet are seamlessly applied to these models using IDM-VTON, leveraging its advanced diffusion-based approach to ensure authenticity and detail preservation. To further refine the visuals, image resolution is enhanced through the jasperai/Flux.1-dev-ControlNet-Upscaler, creating outputs that meet real-world expectations. This article delves into the strengths and gaps identified in try-on performance, offering insights into the future potential of virtual try-on systems.

<div style="display: flex; justify-content: space-around;">
  <img src="./images/generated_images/uncensored_image_hf-2.png" alt="Gen Image 1" width="48%">
  <img src="./images/generated_images/uncensored_image_hf-3.png" alt="Gen Image 2" width="48%">
    <img src="./images/generated_images/uncensored_image_hf-4.png" alt="Gen Image 3" width="48%">
  <img src="./images/generated_images/uncensored_image_hf-10.png" alt="Gen Image 4" width="48%">
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
