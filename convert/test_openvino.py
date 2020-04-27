import os
import cv2
import sys
import logging as log
import numpy as np
from time import time
from openvino.inference_engine import IENetwork, IECore
from efficient.Im_prepro import preprocess_imgs

if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    model_dir = "./openvino_model"
    model_name = "panel"
    image_dir = "C:\Users\Tina_VI01\Desktop\KelvinWu"
    cpu_extension = None
    device = "CPU"
    number_top = 1
    labels = "./imagenet.labels"
    
    model_xml = os.path.join(model_dir, model_name + ".xml")
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")
    
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    
    #supported_layers = ie.query_network(net, device)
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
    net.batch_size = len(os.listdir(image_dir)) #!
    #print(net.batch_size)
    
    # Read and pre-process input images
    n, h, w, c = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, h, w, c))
    for i in range(len(os.listdir(image_dir))):        
        image_path = os.path.join(image_dir, os.listdir(image_dir)[i])
        imag_bgr_small, imag_c_small = preprocess_imgs(image_path) #3/24 revised
        image = imag_c_small.reshape(-1,h,w,c)
        image = image.astype('float32') / 255
        images[i] = image
    
    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    t0 = time()
    res = exec_net.infer(inputs={input_blob: images})
    total_time = time() - t0
    print("Time Elapsed: {:.4f} sec./pic".format(total_time/len(os.listdir(image_dir))))
    
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
        print("Image {}\n".format(os.listdir(image_dir)[i]))
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

        