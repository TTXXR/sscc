import csv
from utils.utils import get_norm

# from model.nsm.net import *
# from model.nsm.my_config import conf

from model.mlp.config import conf
from model.mlp.net import *


class Server(object):
    def __init__(self, model_path, epoch, batch_size=1):
        print("Initializing model...")
        conf["model"]["batch_size"] = batch_size
        conf["model"]["load_path"] = model_path
        conf["model"]["save_path"] = model_path
        conf["model"]["train_source"] = None
        conf["model"]["test_source"] = None

        self.model = Model(**conf["model"])
        print("Model initialization complete.")

        self.model.load(model_path, epoch)
        self.model.to_eval()

        self.data = torch.empty(0, 5307)
        self.input_mean, self.input_std = get_norm("C:/Users/rr/Desktop/documents/Export/InputNorm.txt")
        self.output_mean, self.output_std = get_norm("C:/Users/rr/Desktop/documents/Export/OutputNorm.txt")

        self.csv_writer = csv.writer(open('test.csv', 'w', newline=""))

    def forward(self, x):

        torch.cuda.empty_cache()
        with torch.no_grad():

            x = torch.tensor(x)
            x = (x - self.input_mean) / self.input_std
            data = self.model.forward(x.unsqueeze(0))
            data = data[0].cpu().detach()
            data = data * self.output_std + self.output_mean
            data = data.numpy().tolist()
            self.csv_writer.writerow(data)
        return data
