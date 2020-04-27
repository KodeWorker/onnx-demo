import os
import sys
import logging as log
import numpy as np
from time import time
from PIL import Image
from openvino.inference_engine import IENetwork, IECore

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[-1]):
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 224, 224, 3).astype('float32')
    return norm_img_data

if __name__ == "__main__":
    n_times = 100
    for model_name in ['efficientnet-b7']:
        
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        model_dir = "./openvino_models/{}".format(model_name)
        image_path = r"C:\Users\Tina_VI01\Desktop\KelvinWu\cat.1.jpg"
        cpu_extension = None
        device = "CPU"
        number_top = 1
        labels = "./imagenet.labels"
        
        model_xml = os.path.join(model_dir, "opt-" + model_name + ".xml")
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        log.info("Creating Inference Engine")
        ie = IECore()
        if cpu_extension and 'CPU' in device:
            ie.add_extension(cpu_extension, "CPU")
        
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = IENetwork(model=model_xml, weights=model_bin)
        
        supported_layers = ie.query_network(net, device)
        if "CPU" == device:
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)
        
        #if 'GPU' == device:
        #    supported_layers.update(ie.query_network(net, 'CPU'))
        #    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]

        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        
        log.info("Preparing input blobs")
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        #print(net.batch_size)
        
        # Read and pre-process input images
        x = Image.open(image_path)
        x = x.resize((224, 224))
        #x = preprocess(np.uint8(x))
        x = np.uint8(x)
        x = np.expand_dims(x, axis=0)
        x = np.moveaxis(x, -1, 1)
        
        # Loading model to the plugin
        log.info("Loading model to the plugin")
        exec_net = ie.load_network(network=net, device_name=device)

        # Start sync inference
        log.info("Starting inference in synchronous mode")
        t1 = 0
        for _ in range(n_times):
            t0 = time()
            res = exec_net.infer(inputs={input_blob: x})
            t1 += time() - t0
        log.info("[{}] Elapsed Time: {:4f} sec.\n".format(model_name, t1/n_times))
 
        # Processing output blob
        log.info("Processing output blob")
        res = res[out_blob]
        log.info("Top {} results: ".format(number_top))
        if labels:
            with open(labels, 'r') as f:
                labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        else:
            labels_map = None
        classid_str = "classid"
        probability_str = "probability"
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-number_top:][::-1]
            print(classid_str, probability_str)
            print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
            for id in top_ind:
                det_label = labels_map[id] if labels_map else "{}".format(id)
                label_length = len(det_label)
                space_num_before = (len(classid_str) - label_length) // 2
                space_num_after = len(classid_str) - (space_num_before + label_length) + 2
                space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
                print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                              ' ' * space_num_after, ' ' * space_num_before_prob,
                                              probs[id]))
            print("\n")
        log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")
