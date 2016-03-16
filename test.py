class regressionlienaire():

    def __init__(self, alpha):
        self.alpha = alpha

    def prediction(self):
        return 2342


model1 = regressionlienaire(alpha=2)
model2 = regressionlienaire(alpha=10)
model3 = regressionlienaire(alpha=100)

list = [model1, model2, model3]


for model in list:
    print(model.alpha)
    model.alpha = 2000



