import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision.transforms import transforms


class OnnxRuntime:
    def __init__(self, model) -> None:
        providers = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model, providers=providers)
        
        self.transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
                            
    def __call__(self, data):
        return self.onnxprocess(data)
    
    def onnxprocess(self, image):
        image = self.input_process(image)
        ort_inputs = {self.ort_session.get_inputs()[0].name: image.numpy()}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        return self.softmax_stable(ort_outputs[0][0])

    # def resize(self, image):
    #     image = image.resize((256, 256))
    #     return np.array(image)/ 127.0

    # def normalize(self, image):
    #     mean = np.array([0.485, 0.456, 0.406]).reshape(-1,1,1)
    #     std = np.array([0.229, 0.224, 0.225]).reshape(-1,1,1)
    #     return (image - mean) / std

    
    # def to_tensor(self, image):
    #     image = image.transpose((2, 0, 1))
    #     image = np.expand_dims(image, axis=0)
    #     return image.astype(np.float32)
    
    # def processing(self, image):
    #     image = self.resize(image)
    #     image = self.to_tensor(image)
    #     image = self.normalize(image)
    #     return image.astype(np.float32)
    
    def input_process(self, img):
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img
    
    
    def softmax_stable(self, x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

# import torch   
    
# image_dir = '/home/dev/Xavier/img_classify_shortdate/dataset/valid/partial_lined_merged_cells/1683887178096900.jpg'

# a = OnnxRuntime('/home/dev/Xavier/img_classify_shortdate/onnx_table.onnx')
# img = Image.open(image_dir)
# result = a(img)

# print(result)