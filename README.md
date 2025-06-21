# MNIST Handwritten Digit Generator

This project contains a simple conditional GAN (cGAN) trained from scratch on the MNIST dataset and a Streamlit web app to generate handwritten digits.

## Training
Run the training script to create `mnist_cgan_generator.pth`:

```bash
python mnist_gan_training.py
```

Training requires a GPU for reasonable speed and downloads the MNIST dataset automatically.

## Web App
After training, launch the Streamlit app:

```bash
streamlit run app.py
```

Select a digit (0-9) and click **Generate 5 Images** to see examples from the trained generator.

Deploy the app to Streamlit Community Cloud or similar platforms by uploading `app.py`, `requirements.txt`, and the saved model weights.
