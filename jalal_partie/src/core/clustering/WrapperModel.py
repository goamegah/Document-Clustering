class WrapperModel:
    def __init__(self,model_name:str,requirements:dict[str,object]):
        self.model=globals()[model_name](**requirements)
        self.model_name=model_name

    def get_nmi(self):
        return self.model.get_nmi()

    def get_ari(self):
        return self.model.get_ari()












