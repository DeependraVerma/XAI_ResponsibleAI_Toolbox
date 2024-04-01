import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from skimage.segmentation import slic
import matplotlib.pyplot as plt

class Explainability:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def shap_explanation(self, X):
        explainer = shap.Explainer(self.model, self.data)
        shap_values = explainer.shap_values(X)
        return shap_values

    def lime_explanation(self, X):
        explainer = LimeTabularExplainer(self.data, mode='regression', feature_names=self.data.columns)
        exp = explainer.explain_instance(X.values[0], self.model.predict)
        return exp.as_list()

    def visualize_slic(self, image_path):
        image = plt.imread(image_path)
        segments_slic = slic(image, n_segments=100, compactness=10, sigma=1)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(segments_slic, cmap='viridis')
        ax[1].set_title('SLIC Segments')
        ax[1].axis('off')
        plt.show()
