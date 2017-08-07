
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
        im = datasource.next_im()
        #print im
        #t = time.time()
        img = datasource.get_image(im)
        gt = datasource.get_ground_truth(im)
        data,label = image_processor.build_data_and_label(img, gt)
        #print time.time() - t
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