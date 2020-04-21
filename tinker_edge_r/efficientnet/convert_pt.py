import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
from efficientnet_pytorch import EfficientNet

PT_MODEL = './efficientnet-b1.pt'
RKNN_MODEL = './efficientnet-b1.rknn'

def export_pytorch_model():
    net = EfficientNet.from_pretrained('efficientnet-b1')
    net.set_swish(memory_efficient=False)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1,3,224,224))
    trace_model.save(PT_MODEL)
    
def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'resnet50v2\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)

if __name__ == '__main__':
    
    model = PT_MODEL
    input_size_list = [[3,224,224]]
    
    export_pytorch_model()
    
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
    print('done')

    # Load pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export efficientnet-b1 failed!')
        exit(ret)
    print('done')

    ret = rknn.load_rknn('./resnet_18.rknn')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()

