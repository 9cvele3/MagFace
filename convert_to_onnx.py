import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import onnx

from inference import network_inf

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='magface_epoch_00025.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--arch', default='iresnet100', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    print(f"model_keys: { list(model_keys)[0:10] }")
    print(f"ckpt_keys: { list(ckpt_keys)[0:10] }")
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Model keys:{}'.format(len(model_keys)))
    print('ckpt keys:{}'.format(len(ckpt_keys)))
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return len(used_pretrained_keys) > 0


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.replace(prefix, "", 1) if x.startswith(prefix) or x.startswith("features." + prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    print(pretrained_dict.keys())
    #if "state_dict" in pretrained_dict.keys():
    #    print(list(pretrained_dict['state_dict'].keys())[0:10])
    #    pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'features.')
    #else:
    #    pretrained_dict = remove_prefix(pretrained_dict, 'features.')

    if "state_dict" in pretrained_dict.keys():
        print(list(pretrained_dict['state_dict'].keys())[0:10])
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    print(list(pretrained_dict.keys())[0:10])
    check_keys(model, pretrained_dict)

    # size mismatch for fc.weight: copying a param with shape torch.Size([512, 10572]) from checkpoint,
    # the shape in current model is torch.Size([512, 25088]).
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # net and model
    net = network_inf.NetworkBuilder_inf(args)

    print(args.cpu)
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cuda")
    net = net.to(device)


    # ------------------------ export -----------------------------
    batch_size = 1
    output_onnx = args.trained_model + '.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    x = torch.randn(batch_size, 3, 112, 112).to(device)
    net = net
    torch_out = net(x)
    input_names = ["input"]
    output_names = ["output"]
    #inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)

    torch_out = torch.onnx.export(net,
                                   x,
                                   output_onnx,
                                   export_params=True,
                                   opset_version=9,
                                   verbose=True,
                                   do_constant_folding=True,
                                   input_names=input_names,
                                   output_names=output_names,
                                   dynamic_axes={
                                          'input' : {0 : 'batch_size'},
                                          'output' : {0 : 'batch_size'},
                                        },
                                   keep_initializers_as_inputs=True)

    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)