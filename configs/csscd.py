from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 33
record_dir = 'data/model/LTH+SS+FZJTJW+FZLBJW+HLJ'
train.batch_size = 8
test.batch_size = 4
train.epoch = 300
train.dataset = 'StrokeExtraction_train'

test.dataset = 'StrokeExtraction_val'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test