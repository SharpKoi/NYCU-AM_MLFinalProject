# NYCU AM - ML Final Project 

Topic: [Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/)  

Complete project with codes and all required data is on my [Google Drive](https://drive.google.com/drive/folders/13-POqZxMWREzEseudT7FQCPh_vFBPDsx?usp=sharing).

## Source Codes

- **Main notebook** 

  The codes for main processes like data cleaning, data preprocessing and model training/evaluation/prediction. Which is contained in this repository. See [here](https://github.com/SharpKoi/NYCU-AM_MLFinalProject/blob/master/ML%20Final%20Project%20-%20Disaster%20Tweets.ipynb).

- **Utilities**

  The toolkits used in this project like tokenization and weight loading. Which is contained in this repository. See [here](https://github.com/SharpKoi/NYCU-AM_MLFinalProject/tree/master/myutils).

- **SproutNet**

  The codes for neural network implementation. Which is placed in my another project. See [here](https://github.com/SharpKoi/SproutNet).

## External Resources

- Required
  - Training data [[Download](https://drive.google.com/file/d/1dsCzV64-nh5KDpoQ3TmViCTKPF0ZZfIh/view?usp=sharing)]
  - Testing data [[Download](https://drive.google.com/file/d/1rUf1aj_ykiwHCc-3sI_Et8XHRy15_eEy/view?usp=sharing)]
  - GloVe pre-trained word vectors [[Download](https://drive.google.com/file/d/1Nx8uG-ST2n0YPqj0z_E7b_QOlwSNXGqK/view?usp=sharing)]
- Optional

## Usage

Follow the instructions below to run this project.

### From Google Drive

The most simple way is copying the complete project from [Google Drive](https://drive.google.com/drive/folders/13-POqZxMWREzEseudT7FQCPh_vFBPDsx?usp=sharing) to your space and run the [notebook](https://drive.google.com/file/d/1Gnzm50qFCpsYh6PMtWeTBZZXOKra4g0A/view?usp=sharing) on Google Colab. Or you can download them to run locally.

### From Source Codes

1. Clone the repository to your local machine.

   ```shell
   git clone https://github.com/SharpKoi/NYCU-AM_MLFinalProject.git
   ```

2. Download GloVe from [here](https://drive.google.com/file/d/1Nx8uG-ST2n0YPqj0z_E7b_QOlwSNXGqK/view?usp=sharing). And move the GloVe file to `NYCU-AM_MLFinalProject/data/`.

3. Clone sproutnet from github.

   ```shell
   git clone https://github.com/SharpKoi/SproutNet.git
   ```

4. Run the commands to install SproutNet.

   ```shell
   python SproutNet/setup.py build
   python SproutNet/setup.py install
   ```

5. Run the notebook `NYCU-AM_MLFinalProject/ML Final Project - Disaster Tweets.ipynb`.
