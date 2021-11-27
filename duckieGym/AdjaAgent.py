from duckieGym.detector import preprocess_image


class AdjaAgent:

    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        x = preprocess_image(obs)
        return self.model.predict(x)
