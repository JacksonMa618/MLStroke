from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 25
record_dir = 'data/model/CCSE-HW'
model_dir = 'data/model/CCSE-HW'
train.batch_size = 16
test.batch_size = 16
train.epoch = 300
train.dataset = 'ccseHW_train'

test.dataset = 'SCUT_Stroke_test'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test