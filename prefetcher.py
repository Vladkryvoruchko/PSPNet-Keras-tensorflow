from multiprocessing import Pool, cpu_count

import image_processor

class PreFetcher:
    def __init__(self, datasource, batch_size=1, ahead=4):

        cpus = cpu_count()
        self.pool = Pool(processes=min(ahead, cpus))
        self.batch_size = batch_size
        self.ahead = ahead
        self.batch_queue = []

        self.datasource = datasource

    def fetch_batch(self):
        try:
            self.refill_tasks()
            result = self.batch_queue.pop(0)
            return result.get(31536000)[0]
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            self.pool.terminate()
            raise

    def refill_tasks(self):
        while len(self.batch_queue) < self.ahead:
            im = self.datasource.next_im()
            d = (self.datasource, im)

            batch = self.pool.map_async(build_train, [d])
            self.batch_queue.append(batch)

def build_train(d):
    datasource, im = d
    img = datasource.get_image(im)
    gt = datasource.get_ground_truth(im)
    batch = image_processor.build_data_and_label(img, gt)
    return batch

