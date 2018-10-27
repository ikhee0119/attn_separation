class Trainer:
    def __init__(self, train_loader, config):

        self.train_loader = train_loader

        self.num_layers = config.num_layers
        self.input_length = config.input_length

        self.build_model()

    def build_model(self):

    def train(self):
        pass