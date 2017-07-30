

def debug(self, data):
    names = [layer.name for layer in self.model.layers]
    for name in names[-10:]:
        print_activation(self.model, name, data)



def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print layer_name, array_to_str(io)
    if layer_name == "concatenate_1":
        print "Saving", layer_name
        with h5py.File("keras.h5", 'w') as f:
            f.create_dataset('a', data=io)
def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a), np.max(a), np.mean(a))