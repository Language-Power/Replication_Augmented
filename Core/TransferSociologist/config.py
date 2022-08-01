"""
Builds a training config
"""
import os 
import json

class GlobalConfig:
    """
    Config here
    Global config because for the moment it will 
    be model+dataset+training+prediction params
    """
    def __init__(self):
        self.model = {"nlabels": None,
                      "language": None,
                      "bert_model": None,
                      "task_type": None
                    }
        self.data = {"data_path": None,
                     "data_type": None, 
                     "test_size": None,
                     "random_seed": None,
                     "downsampling": None,
                     "gold_standard_path": None,
                     "state_treated": None 
                    }
        self.train = {"nepochs": None,
                      "batch_size": None,
                      "learning_rate": None,
                      "worker": None,
                      "save_model": None,
                      "path_to_model": None
                      }
        self.predictor = {"path_to_model": None,
                          "batch_size": None}

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.data = json.loads(lines[0])
            self.model = json.loads(lines[1])
            self.train = json.loads(lines[2])
            self.predictor = json.loads(lines[3])
    def save(self, path):
        """Here we save the file as a jsonl file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f)
            f.write('\n')
            json.dump(self.model, f)
            f.write('\n')
            json.dump(self.train, f)
            f.write('\n')
            json.dump(self.predictor, f)
            f.write('\n')

#    def modif(self, dic):

class BaseConfig(GlobalConfig):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.model.update({"nlabels": 2,
                      "bert_model": "Camembert"
                    })
        self.data.update({"data_type": "doccanno", 
                     "test_size": 0.2,
                     "random_seed": 2020,
                     "downsampling": False,
                     "state_treated": False
                    })
        self.train.update({"nepochs": 20,
                      "batch_size": 32,
                      "learning_rate": 1e-5,
                      "worker": "cuda",
                      "save_model": False,
                      })

class BaseTextClassifConfig(BaseConfig):
    def __init__(self):
        super(BaseTextClassifConfig, self).__init__()
        self.model['task_type'] = 'text_classification'

class BaseSentClassifConfig(BaseConfig):
    def __init__(self):
        super(BaseSentClassifConfig, self).__init__()
        self.model['task_type'] = 'sentence_classification'

class BaseSeqLabConfig(BaseConfig):
    def __init__(self):
        super(BaseSeqLabConfig, self).__init__()
        self.model['task_type'] = 'sequence_labelling'
