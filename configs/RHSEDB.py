from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 10
record_dir = 'data/model/RHSEDB'
train.batch_size = 8
test.batch_size = 4
train.epoch = 180
train.dataset = 'RHSEDB_train'

test.dataset = 'RHSEDB_test'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test