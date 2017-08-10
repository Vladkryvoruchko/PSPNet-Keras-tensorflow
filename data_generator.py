import time
import threading

import image_processor

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def DataGenerator(datasource):
    while True:
        # t = time.time()
        im = datasource.next_im()
        img = datasource.get_image(im)
        gt = datasource.get_ground_truth(im)
        outs = image_processor.scale_and_crop([img, gt])

        data = outs[0]
        label = outs[1]

        # Batch size of 1
        data = data[np.newaxis, ...]
        label = label[np.newaxis, ...]

        #print time.time() - t
        yield (data,label)

@threadsafe_generator
def DiscGenerator(datasource):
    while True:
        im = datasource.next_im()
        # print im
        # t = time.time()
        img = datasource.get_image(im)
        gt = datasource.get_ground_truth(im)
        pred = datasource.get_prediction(im, threshold=0.5)

        outs = image_processor.scale_and_crop([img,gt,pred])
        img = outs[0]
        gt = outs[1]
        pred = outs[2]

        # Select only one category
        c = 1
        gt = gt[c-1]
        pred = pred[c-1]
        data1 = np.concatenate((data,gt), axis=-1)
        data2 = np.concatenate((data,pred), axis=-1)
        label1 = np.array([1,0])
        label2 = np.array([0,1])
        data = np.stack((data1,data2))
        label = np.stack((label1,label2))

        print data.shape, label.shape
        # print time.time() - t
        yield (data,label)

        
'''
g = data_generator(datasource)
c = 0
print self.model.metrics_names
for x,y in g:
    output(x,y,prefix=str(c)+"_")
    print self.model.train_on_batch(x,y)
    print self.model.test_on_batch(x,y)
    pred = self.model.predict_on_batch(x)
    output(x,pred,prefix=str(c)+"__", slice_y=True)
    c += 1
    if c == 3:
        raise
'''
